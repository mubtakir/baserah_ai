#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Guided Video Analyzer - Part 2: Visual Video Analysis
محلل الفيديو الموجه بالخبير - الجزء الثاني: تحليل الفيديو البصري

Revolutionary integration of Expert/Explorer guidance with video analysis,
applying adaptive mathematical equations to enhance video understanding.

التكامل الثوري لتوجيه الخبير/المستكشف مع تحليل الفيديو،
تطبيق المعادلات الرياضية المتكيفة لتحسين فهم الفيديو.

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

# محاكاة النظام المتكيف للفيديو
class MockVideoEquation:
    def __init__(self, name: str, input_dim: int, output_dim: int):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_complexity = 12  # الفيديو أكثر تعقيداً من الصور
        self.adaptation_count = 0
        self.video_accuracy = 0.65  # الفيديو أصعب في التحليل
        self.motion_smoothness = 0.75
        self.temporal_consistency = 0.8
        self.frame_quality = 0.7
        self.narrative_flow = 0.85

    def adapt_with_expert_guidance(self, guidance, analysis):
        self.adaptation_count += 1
        if hasattr(guidance, 'recommended_evolution'):
            if guidance.recommended_evolution == "increase":
                self.current_complexity += 3
                self.video_accuracy += 0.04
                self.motion_smoothness += 0.03
                self.temporal_consistency += 0.02
            elif guidance.recommended_evolution == "restructure":
                self.video_accuracy += 0.02
                self.frame_quality += 0.04
                self.narrative_flow += 0.03

    def get_expert_guidance_summary(self):
        return {
            "current_complexity": self.current_complexity,
            "total_adaptations": self.adaptation_count,
            "video_accuracy": self.video_accuracy,
            "motion_smoothness": self.motion_smoothness,
            "temporal_consistency": self.temporal_consistency,
            "frame_quality": self.frame_quality,
            "narrative_flow": self.narrative_flow,
            "average_improvement": 0.1 * self.adaptation_count
        }

class MockVideoGuidance:
    def __init__(self, target_complexity, focus_areas, adaptation_strength, priority_functions, recommended_evolution):
        self.target_complexity = target_complexity
        self.focus_areas = focus_areas
        self.adaptation_strength = adaptation_strength
        self.priority_functions = priority_functions
        self.recommended_evolution = recommended_evolution

class MockVideoAnalysis:
    def __init__(self, video_accuracy, motion_quality, temporal_stability, frame_consistency, narrative_coherence, areas_for_improvement):
        self.video_accuracy = video_accuracy
        self.motion_quality = motion_quality
        self.temporal_stability = temporal_stability
        self.frame_consistency = frame_consistency
        self.narrative_coherence = narrative_coherence
        self.areas_for_improvement = areas_for_improvement

@dataclass
class VideoAnalysisRequest:
    """طلب تحليل الفيديو"""
    shape: ShapeEntity
    analysis_type: str  # "motion", "temporal", "narrative", "quality", "comprehensive"
    video_aspects: List[str]  # ["smoothness", "consistency", "flow", "transitions"]
    expert_guidance_level: str = "adaptive"
    learning_enabled: bool = True
    motion_optimization: bool = True

@dataclass
class VideoAnalysisResult:
    """نتيجة تحليل الفيديو"""
    success: bool
    video_compliance: Dict[str, float]
    motion_violations: List[str]
    video_insights: List[str]
    motion_metrics: Dict[str, float]
    temporal_analysis: Dict[str, Any]
    frame_quality_scores: Dict[str, float]
    narrative_evaluation: Dict[str, float]
    expert_guidance_applied: Dict[str, Any] = None
    equation_adaptations: Dict[str, Any] = None
    performance_improvements: Dict[str, float] = None
    learning_insights: List[str] = None
    next_cycle_recommendations: List[str] = None

class ExpertGuidedVideoAnalyzer:
    """محلل الفيديو الموجه بالخبير الثوري"""

    def __init__(self):
        """تهيئة محلل الفيديو الموجه بالخبير"""
        print("🌟" + "="*90 + "🌟")
        print("🎬 محلل الفيديو الموجه بالخبير الثوري")
        print("🎥 الخبير/المستكشف يقود التحليل السينمائي بذكاء")
        print("🧮 معادلات رياضية متكيفة + تحليل فيديو متقدم")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*90 + "🌟")

        # إنشاء معادلات الفيديو متخصصة
        self.video_equations = {
            "motion_analyzer": MockVideoEquation("motion_analysis", 16, 12),
            "temporal_consistency_checker": MockVideoEquation("temporal_consistency", 14, 10),
            "frame_quality_evaluator": MockVideoEquation("frame_quality", 12, 8),
            "transition_smoother": MockVideoEquation("transition_smoothing", 18, 14),
            "narrative_flow_assessor": MockVideoEquation("narrative_flow", 20, 16),
            "scene_coherence_detector": MockVideoEquation("scene_coherence", 15, 11),
            "visual_continuity_tracker": MockVideoEquation("visual_continuity", 13, 9),
            "pacing_optimizer": MockVideoEquation("pacing_optimization", 11, 7),
            "cinematic_quality_meter": MockVideoEquation("cinematic_quality", 17, 13)
        }

        # معايير تحليل الفيديو
        self.video_standards = {
            "motion_smoothness": {
                "name": "سلاسة الحركة",
                "criteria": "انسيابية وطبيعية الحركة",
                "spiritual_meaning": "الحركة المتناغمة تعكس النظام الإلهي"
            },
            "temporal_consistency": {
                "name": "الاتساق الزمني",
                "criteria": "ثبات العناصر عبر الزمن",
                "spiritual_meaning": "الثبات على المبادئ سنة إلهية"
            },
            "narrative_flow": {
                "name": "تدفق السرد",
                "criteria": "تسلسل منطقي للأحداث",
                "spiritual_meaning": "الحكمة في ترتيب الأمور"
            },
            "visual_harmony": {
                "name": "التناغم البصري",
                "criteria": "انسجام العناصر البصرية",
                "spiritual_meaning": "الجمال انعكاس للكمال الإلهي"
            }
        }

        # تاريخ التحليلات السينمائية
        self.video_history = []
        self.video_learning_database = {}

        print("🎬 تم إنشاء المعادلات السينمائية المتخصصة:")
        for eq_name in self.video_equations.keys():
            print(f"   ✅ {eq_name}")

        print("✅ تم تهيئة محلل الفيديو الموجه بالخبير!")

    def analyze_video_with_expert_guidance(self, request: VideoAnalysisRequest) -> VideoAnalysisResult:
        """تحليل الفيديو موجه بالخبير"""
        print(f"\n🎬 بدء تحليل الفيديو الموجه بالخبير لـ: {request.shape.name}")
        start_time = datetime.now()

        # المرحلة 1: تحليل الخبير للطلب السينمائي
        expert_analysis = self._analyze_video_request_with_expert(request)
        print(f"🎥 تحليل الخبير السينمائي: {expert_analysis['complexity_assessment']}")

        # المرحلة 2: توليد توجيهات الخبير للمعادلات السينمائية
        expert_guidance = self._generate_video_expert_guidance(request, expert_analysis)
        print(f"🎬 توجيه الخبير السينمائي: {expert_guidance.recommended_evolution}")

        # المرحلة 3: تكيف المعادلات السينمائية
        equation_adaptations = self._adapt_video_equations(expert_guidance, expert_analysis)
        print(f"🧮 تكيف المعادلات السينمائية: {len(equation_adaptations)} معادلة")

        # المرحلة 4: تنفيذ التحليل السينمائي المتكيف
        video_analysis = self._perform_adaptive_video_analysis(request, equation_adaptations)

        # المرحلة 5: فحص المعايير السينمائية
        video_compliance = self._check_video_standards_compliance(request, video_analysis)

        # المرحلة 6: تحليل الحركة
        motion_metrics = self._analyze_motion_quality(request, video_analysis)

        # المرحلة 7: التحليل الزمني
        temporal_analysis = self._analyze_temporal_consistency(request, video_analysis)

        # المرحلة 8: تقييم جودة الإطارات
        frame_quality_scores = self._evaluate_frame_quality(request, video_analysis)

        # المرحلة 9: تقييم السرد
        narrative_evaluation = self._evaluate_narrative_flow(request, video_analysis)

        # المرحلة 10: قياس التحسينات السينمائية
        performance_improvements = self._measure_video_improvements(request, video_analysis, equation_adaptations)

        # المرحلة 11: استخراج رؤى التعلم السينمائي
        learning_insights = self._extract_video_learning_insights(request, video_analysis, performance_improvements)

        # المرحلة 12: توليد توصيات للدورة التالية
        next_cycle_recommendations = self._generate_video_next_cycle_recommendations(performance_improvements, learning_insights)

        # إنشاء النتيجة السينمائية
        result = VideoAnalysisResult(
            success=True,
            video_compliance=video_compliance["compliance_scores"],
            motion_violations=video_compliance["violations"],
            video_insights=video_analysis["insights"],
            motion_metrics=motion_metrics,
            temporal_analysis=temporal_analysis,
            frame_quality_scores=frame_quality_scores,
            narrative_evaluation=narrative_evaluation,
            expert_guidance_applied=expert_guidance.__dict__,
            equation_adaptations=equation_adaptations,
            performance_improvements=performance_improvements,
            learning_insights=learning_insights,
            next_cycle_recommendations=next_cycle_recommendations
        )

        # حفظ في قاعدة التعلم السينمائي
        self._save_video_learning(request, result)

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ انتهى التحليل السينمائي الموجه في {total_time:.2f} ثانية")

        return result

    def _analyze_video_request_with_expert(self, request: VideoAnalysisRequest) -> Dict[str, Any]:
        """تحليل الطلب السينمائي بواسطة الخبير"""

        # تحليل الخصائص السينمائية للشكل
        motion_complexity = len(request.shape.equation_params) * 2.0
        temporal_richness = len(request.shape.color_properties) * 1.5
        narrative_depth = request.shape.geometric_features.get("area", 100) / 40.0

        # تحليل جوانب الفيديو المطلوبة
        video_aspects_complexity = len(request.video_aspects) * 3.0

        # تحليل نوع التحليل
        analysis_type_complexity = {
            "motion": 3.0,
            "temporal": 3.5,
            "narrative": 4.0,
            "quality": 2.5,
            "comprehensive": 6.0
        }.get(request.analysis_type, 3.0)

        total_video_complexity = motion_complexity + temporal_richness + narrative_depth + video_aspects_complexity + analysis_type_complexity

        return {
            "motion_complexity": motion_complexity,
            "temporal_richness": temporal_richness,
            "narrative_depth": narrative_depth,
            "video_aspects_complexity": video_aspects_complexity,
            "analysis_type_complexity": analysis_type_complexity,
            "total_video_complexity": total_video_complexity,
            "complexity_assessment": "سينمائي معقد" if total_video_complexity > 25 else "سينمائي متوسط" if total_video_complexity > 15 else "سينمائي بسيط",
            "recommended_adaptations": int(total_video_complexity // 5) + 3,
            "focus_areas": self._identify_video_focus_areas(request)
        }

    def _identify_video_focus_areas(self, request: VideoAnalysisRequest) -> List[str]:
        """تحديد مناطق التركيز السينمائي"""
        focus_areas = []

        if "smoothness" in request.video_aspects:
            focus_areas.append("motion_smoothness_enhancement")
        if "consistency" in request.video_aspects:
            focus_areas.append("temporal_consistency_optimization")
        if "flow" in request.video_aspects:
            focus_areas.append("narrative_flow_improvement")
        if "transitions" in request.video_aspects:
            focus_areas.append("transition_quality_enhancement")
        if request.analysis_type == "motion":
            focus_areas.append("motion_analysis_focus")
        if request.analysis_type == "narrative":
            focus_areas.append("narrative_structure_analysis")
        if request.motion_optimization:
            focus_areas.append("motion_optimization")

        return focus_areas

    def _generate_video_expert_guidance(self, request: VideoAnalysisRequest, analysis: Dict[str, Any]):
        """توليد توجيهات الخبير للتحليل السينمائي"""

        # تحديد التعقيد المستهدف للفيديو
        target_complexity = 15 + analysis["recommended_adaptations"]

        # تحديد الدوال ذات الأولوية للتحليل السينمائي
        priority_functions = []
        if "motion_smoothness_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["sin_cos", "gaussian"])
        if "temporal_consistency_optimization" in analysis["focus_areas"]:
            priority_functions.extend(["tanh", "hyperbolic"])
        if "narrative_flow_improvement" in analysis["focus_areas"]:
            priority_functions.extend(["softplus", "swish"])
        if "transition_quality_enhancement" in analysis["focus_areas"]:
            priority_functions.extend(["squared_relu", "softsign"])
        if "motion_analysis_focus" in analysis["focus_areas"]:
            priority_functions.extend(["sin", "cos"])
        if "narrative_structure_analysis" in analysis["focus_areas"]:
            priority_functions.extend(["gaussian", "hyperbolic"])
        if "motion_optimization" in analysis["focus_areas"]:
            priority_functions.extend(["softplus", "swish"])

        # تحديد نوع التطور السينمائي
        if analysis["complexity_assessment"] == "سينمائي معقد":
            recommended_evolution = "increase"
            adaptation_strength = 0.95
        elif analysis["complexity_assessment"] == "سينمائي متوسط":
            recommended_evolution = "restructure"
            adaptation_strength = 0.8
        else:
            recommended_evolution = "maintain"
            adaptation_strength = 0.65

        return MockVideoGuidance(
            target_complexity=target_complexity,
            focus_areas=analysis["focus_areas"],
            adaptation_strength=adaptation_strength,
            priority_functions=priority_functions or ["sin_cos", "tanh"],
            recommended_evolution=recommended_evolution
        )

    def _adapt_video_equations(self, guidance, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تكيف المعادلات السينمائية"""

        adaptations = {}

        # إنشاء تحليل وهمي للمعادلات السينمائية
        mock_analysis = MockVideoAnalysis(
            video_accuracy=0.65,
            motion_quality=0.75,
            temporal_stability=0.8,
            frame_consistency=0.7,
            narrative_coherence=0.85,
            areas_for_improvement=guidance.focus_areas
        )

        # تكيف كل معادلة سينمائية
        for eq_name, equation in self.video_equations.items():
            print(f"   🎬 تكيف معادلة سينمائية: {eq_name}")
            equation.adapt_with_expert_guidance(guidance, mock_analysis)
            adaptations[eq_name] = equation.get_expert_guidance_summary()

        return adaptations

    def _perform_adaptive_video_analysis(self, request: VideoAnalysisRequest, adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ التحليل السينمائي المتكيف"""

        analysis_results = {
            "insights": [],
            "video_calculations": {},
            "motion_predictions": [],
            "cinematic_scores": {}
        }

        # تحليل الحركة
        motion_accuracy = adaptations.get("motion_analyzer", {}).get("video_accuracy", 0.65)
        analysis_results["insights"].append(f"تحليل الحركة السينمائية: دقة {motion_accuracy:.2%}")
        analysis_results["video_calculations"]["motion"] = self._calculate_motion_quality(request.shape)

        # تحليل الاتساق الزمني
        if "temporal" in request.analysis_type:
            temporal_consistency = adaptations.get("temporal_consistency_checker", {}).get("temporal_consistency", 0.8)
            analysis_results["insights"].append(f"الاتساق الزمني: ثبات {temporal_consistency:.2%}")
            analysis_results["video_calculations"]["temporal"] = self._calculate_temporal_metrics(request.shape)

        # تحليل جودة الإطارات
        if "quality" in request.analysis_type:
            frame_quality = adaptations.get("frame_quality_evaluator", {}).get("frame_quality", 0.7)
            analysis_results["insights"].append(f"جودة الإطارات: مستوى {frame_quality:.2%}")
            analysis_results["video_calculations"]["frames"] = self._calculate_frame_quality(request.shape)

        # تحليل السرد
        if "narrative" in request.analysis_type:
            narrative_flow = adaptations.get("narrative_flow_assessor", {}).get("narrative_flow", 0.85)
            analysis_results["insights"].append(f"تدفق السرد: انسيابية {narrative_flow:.2%}")
            analysis_results["video_calculations"]["narrative"] = self._calculate_narrative_flow(request.shape)

        return analysis_results

    def _calculate_motion_quality(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب جودة الحركة"""
        smoothness = min(1.0, shape.geometric_features.get("area", 100) / 150.0)
        velocity_consistency = len(shape.equation_params) * 0.12
        acceleration_smoothness = min(1.0, velocity_consistency)

        return {
            "smoothness": smoothness,
            "velocity_consistency": min(1.0, velocity_consistency),
            "acceleration_smoothness": acceleration_smoothness,
            "overall_motion_quality": (smoothness + velocity_consistency + acceleration_smoothness) / 3
        }

    def _calculate_temporal_metrics(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب المقاييس الزمنية"""
        consistency = shape.position_info.get("center_x", 0.5) * 1.5
        stability = shape.geometric_features.get("stability", 0.8)
        continuity = min(1.0, shape.geometric_features.get("area", 100) / 120.0)

        return {
            "consistency": min(1.0, consistency),
            "stability": stability,
            "continuity": continuity,
            "temporal_score": (consistency + stability + continuity) / 3
        }

    def _calculate_frame_quality(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب جودة الإطارات"""
        resolution_quality = min(1.0, shape.geometric_features.get("area", 100) / 180.0)
        clarity = len(shape.equation_params) * 0.18
        detail_preservation = min(1.0, clarity)

        return {
            "resolution_quality": resolution_quality,
            "clarity": min(1.0, clarity),
            "detail_preservation": detail_preservation,
            "frame_quality_score": (resolution_quality + clarity + detail_preservation) / 3
        }

    def _calculate_narrative_flow(self, shape: ShapeEntity) -> Dict[str, float]:
        """حساب تدفق السرد"""
        coherence = shape.geometric_features.get("coherence", 0.85)
        pacing = min(1.0, shape.geometric_features.get("area", 100) / 160.0)
        engagement = len(shape.equation_params) * 0.15

        return {
            "coherence": coherence,
            "pacing": pacing,
            "engagement": min(1.0, engagement),
            "narrative_score": (coherence + pacing + engagement) / 3
        }

    def _check_video_standards_compliance(self, request: VideoAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """فحص الامتثال للمعايير السينمائية"""

        compliance = {
            "compliance_scores": {},
            "violations": [],
            "recommendations": []
        }

        # فحص سلاسة الحركة
        motion_data = analysis["video_calculations"].get("motion", {})
        motion_score = motion_data.get("overall_motion_quality", 0.5)
        compliance["compliance_scores"]["motion_smoothness"] = motion_score
        if motion_score < 0.6:
            compliance["violations"].append("حركة غير سلسة")

        # فحص الاتساق الزمني
        if "temporal" in analysis["video_calculations"]:
            temporal_data = analysis["video_calculations"]["temporal"]
            temporal_score = temporal_data.get("temporal_score", 0.5)
            compliance["compliance_scores"]["temporal_consistency"] = temporal_score
            if temporal_score < 0.65:
                compliance["violations"].append("اتساق زمني ضعيف")

        return compliance

    def _analyze_motion_quality(self, request: VideoAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, float]:
        """تحليل جودة الحركة"""
        motion_data = analysis["video_calculations"].get("motion", {})

        return {
            "smoothness_quality": motion_data.get("smoothness", 0.7),
            "velocity_quality": motion_data.get("velocity_consistency", 0.6),
            "acceleration_quality": motion_data.get("acceleration_smoothness", 0.8),
            "overall_motion_quality": motion_data.get("overall_motion_quality", 0.7)
        }

    def _analyze_temporal_consistency(self, request: VideoAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل الاتساق الزمني"""
        temporal_data = analysis["video_calculations"].get("temporal", {})

        return {
            "consistency_analysis": {
                "score": temporal_data.get("consistency", 0.8),
                "stability": temporal_data.get("stability", 0.8),
                "assessment": "ممتاز" if temporal_data.get("consistency", 0.8) > 0.8 else "جيد" if temporal_data.get("consistency", 0.8) > 0.6 else "يحتاج تحسين"
            },
            "continuity_analysis": {
                "level": temporal_data.get("continuity", 0.75),
                "effectiveness": "فعال" if temporal_data.get("continuity", 0.75) > 0.7 else "متوسط"
            },
            "overall_temporal_score": temporal_data.get("temporal_score", 0.75)
        }

    def _evaluate_frame_quality(self, request: VideoAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, float]:
        """تقييم جودة الإطارات"""
        frame_data = analysis["video_calculations"].get("frames", {})

        return {
            "resolution_score": frame_data.get("resolution_quality", 0.8),
            "clarity_score": frame_data.get("clarity", 0.7),
            "detail_score": frame_data.get("detail_preservation", 0.75),
            "overall_frame_quality": frame_data.get("frame_quality_score", 0.75)
        }

    def _evaluate_narrative_flow(self, request: VideoAnalysisRequest, analysis: Dict[str, Any]) -> Dict[str, float]:
        """تقييم تدفق السرد"""
        narrative_data = analysis["video_calculations"].get("narrative", {})

        return {
            "coherence_score": narrative_data.get("coherence", 0.85),
            "pacing_score": narrative_data.get("pacing", 0.7),
            "engagement_score": narrative_data.get("engagement", 0.6),
            "overall_narrative_quality": narrative_data.get("narrative_score", 0.7)
        }

    def _measure_video_improvements(self, request: VideoAnalysisRequest, analysis: Dict[str, Any], adaptations: Dict[str, Any]) -> Dict[str, float]:
        """قياس تحسينات الفيديو"""

        improvements = {}

        # تحسن جودة الحركة
        avg_motion = np.mean([adapt.get("motion_smoothness", 0.75) for adapt in adaptations.values()])
        baseline_motion = 0.6
        motion_improvement = ((avg_motion - baseline_motion) / baseline_motion) * 100
        improvements["motion_quality_improvement"] = max(0, motion_improvement)

        # تحسن الاتساق الزمني
        avg_temporal = np.mean([adapt.get("temporal_consistency", 0.8) for adapt in adaptations.values()])
        baseline_temporal = 0.65
        temporal_improvement = ((avg_temporal - baseline_temporal) / baseline_temporal) * 100
        improvements["temporal_improvement"] = max(0, temporal_improvement)

        # تحسن جودة الإطارات
        avg_frame = np.mean([adapt.get("frame_quality", 0.7) for adapt in adaptations.values()])
        baseline_frame = 0.55
        frame_improvement = ((avg_frame - baseline_frame) / baseline_frame) * 100
        improvements["frame_improvement"] = max(0, frame_improvement)

        # تحسن السرد
        avg_narrative = np.mean([adapt.get("narrative_flow", 0.85) for adapt in adaptations.values()])
        baseline_narrative = 0.7
        narrative_improvement = ((avg_narrative - baseline_narrative) / baseline_narrative) * 100
        improvements["narrative_improvement"] = max(0, narrative_improvement)

        return improvements

    def _extract_video_learning_insights(self, request: VideoAnalysisRequest, analysis: Dict[str, Any], improvements: Dict[str, float]) -> List[str]:
        """استخراج رؤى التعلم السينمائي"""

        insights = []

        if improvements["motion_quality_improvement"] > 20:
            insights.append("التكيف الموجه بالخبير حسن جودة الحركة بشكل ملحوظ")

        if improvements["temporal_improvement"] > 18:
            insights.append("المعادلات المتكيفة ممتازة لتحسين الاتساق الزمني")

        if improvements["frame_improvement"] > 25:
            insights.append("النظام نجح في تحسين جودة الإطارات بشكل كبير")

        if improvements["narrative_improvement"] > 15:
            insights.append("التوجيه الخبير فعال في تحسين تدفق السرد")

        if request.analysis_type == "comprehensive":
            insights.append("التحليل الشامل يستفيد من جميع جوانب النظام الموجه")

        return insights

    def _generate_video_next_cycle_recommendations(self, improvements: Dict[str, float], insights: List[str]) -> List[str]:
        """توليد توصيات للدورة التالية"""

        recommendations = []

        avg_improvement = np.mean(list(improvements.values()))

        if avg_improvement > 22:
            recommendations.append("الحفاظ على إعدادات التكيف السينمائي الحالية")
            recommendations.append("تجربة تحليل سينمائي أكثر تعقيداً")
            recommendations.append("إضافة تحليل المؤثرات البصرية")
        elif avg_improvement > 15:
            recommendations.append("زيادة قوة التكيف السينمائي تدريجياً")
            recommendations.append("إضافة معايير سينمائية جديدة")
            recommendations.append("تحسين دقة تحليل الحركة")
        else:
            recommendations.append("مراجعة استراتيجية التوجيه السينمائي")
            recommendations.append("تحسين دقة المعادلات السينمائية")
            recommendations.append("إعادة تقييم معايير الجودة")

        return recommendations

    def _save_video_learning(self, request: VideoAnalysisRequest, result: VideoAnalysisResult):
        """حفظ التعلم السينمائي"""

        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "shape_name": request.shape.name,
            "analysis_type": request.analysis_type,
            "video_aspects": request.video_aspects,
            "success": result.success,
            "performance_improvements": result.performance_improvements,
            "learning_insights": result.learning_insights
        }

        shape_key = f"{request.shape.category}_{request.analysis_type}"
        if shape_key not in self.video_learning_database:
            self.video_learning_database[shape_key] = []

        self.video_learning_database[shape_key].append(learning_entry)

        # الاحتفاظ بآخر 5 إدخالات
        if len(self.video_learning_database[shape_key]) > 5:
            self.video_learning_database[shape_key] = self.video_learning_database[shape_key][-5:]

def main():
    """اختبار محلل الفيديو الموجه بالخبير"""
    print("🧪 اختبار محلل الفيديو الموجه بالخبير...")

    # إنشاء المحلل
    video_analyzer = ExpertGuidedVideoAnalyzer()

    # إنشاء شكل اختبار
    from revolutionary_database import ShapeEntity

    test_shape = ShapeEntity(
        id=2, name="فيديو سينمائي رائع", category="سينما",
        equation_params={"motion": 0.9, "flow": 0.85, "drama": 0.95, "rhythm": 0.8},
        geometric_features={"area": 200.0, "stability": 0.9, "coherence": 0.88, "uniqueness": 0.92},
        color_properties={"primary_palette": [255, 100, 50], "secondary_palette": [50, 150, 255], "accent_color": [255, 255, 100]},
        position_info={"center_x": 0.5, "center_y": 0.5},
        tolerance_thresholds={}, created_date="", updated_date=""
    )

    # طلب تحليل سينمائي شامل
    analysis_request = VideoAnalysisRequest(
        shape=test_shape,
        analysis_type="comprehensive",
        video_aspects=["smoothness", "consistency", "flow", "transitions"],
        expert_guidance_level="adaptive",
        learning_enabled=True,
        motion_optimization=True
    )

    # تنفيذ التحليل
    result = video_analyzer.analyze_video_with_expert_guidance(analysis_request)

    print(f"\n🎬 نتائج التحليل السينمائي:")
    print(f"   ✅ النجاح: {result.success}")
    if result.success:
        print(f"   🎥 رؤى سينمائية: {len(result.video_insights)} رؤية")
        print(f"   🏃 مقاييس الحركة: {len(result.motion_metrics)} مقياس")
        print(f"   ⏰ التحليل الزمني: متاح")
        print(f"   🖼️ جودة الإطارات: متاح")
        print(f"   📖 تقييم السرد: متاح")

    print(f"\n📊 إحصائيات المحلل:")
    print(f"   🎬 معادلات سينمائية: {len(video_analyzer.video_equations)}")
    print(f"   📚 قاعدة التعلم: {len(video_analyzer.video_learning_database)} إدخال")

if __name__ == "__main__":
    main()