#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Drawing-Extraction Unit with Expert/Explorer Bridge
الوحدة المتكاملة للرسم والاستنباط مع جسر الخبير/المستكشف

This integrated unit combines drawing and extraction capabilities with an Expert/Explorer
bridge to create a feedback loop that continuously improves accuracy and performance.

هذه الوحدة المتكاملة تجمع قدرات الرسم والاستنباط مع جسر الخبير/المستكشف
لإنشاء حلقة تغذية راجعة تحسن الدقة والأداء باستمرار.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revolutionary_database import ShapeEntity
from revolutionary_drawing_unit import RevolutionaryDrawingUnit
from revolutionary_extractor_unit import RevolutionaryExtractorUnit

# استيراد محلي للاختبار
try:
    from .expert_explorer_bridge import ExpertExplorerBridge, ExplorerFeedback
    from .physics_expert_bridge import PhysicsExpertBridge, PhysicsAnalysisResult, ArtisticPhysicsBalance
except ImportError:
    from expert_explorer_bridge import ExpertExplorerBridge, ExplorerFeedback
    from physics_expert_bridge import PhysicsExpertBridge, PhysicsAnalysisResult, ArtisticPhysicsBalance


class IntegratedDrawingExtractionUnit:
    """الوحدة المتكاملة للرسم والاستنباط مع الخبير/المستكشف"""

    def __init__(self):
        """تهيئة الوحدة المتكاملة"""
        print("🌟" + "="*60 + "🌟")
        print("🔗 الوحدة المتكاملة للرسم والاستنباط")
        print("🧠 مع جسر الخبير/المستكشف الثوري")
        print("🌟 إبداع باسل يحيى عبدالله 🌟")
        print("🌟" + "="*60 + "🌟")

        # تهيئة المكونات
        self.drawing_unit = RevolutionaryDrawingUnit()
        self.extractor_unit = RevolutionaryExtractorUnit()
        self.expert_bridge = ExpertExplorerBridge()

        # تهيئة جسر الخبير الفيزيائي
        self.physics_expert = PhysicsExpertBridge(self.expert_bridge)

        # إحصائيات التكامل
        self.integration_cycles = 0
        self.successful_cycles = 0
        self.improvement_history = []

        print("✅ تم تهيئة الوحدة المتكاملة بنجاح!")

    def execute_integrated_cycle(self, shape: ShapeEntity,
                                learn_from_cycle: bool = True) -> Dict[str, Any]:
        """تنفيذ دورة متكاملة: رسم → استنباط → فيزياء → تحليل → تعلم"""
        print(f"\n🔄 بدء الدورة المتكاملة الفيزيائية لـ: {shape.name}")

        cycle_result = {
            "shape_name": shape.name,
            "category": shape.category,
            "cycle_number": self.integration_cycles + 1,
            "stages": {},
            "physics_analysis": {},
            "artistic_physics_balance": {},
            "overall_success": False,
            "improvements_applied": []
        }

        try:
            # المرحلة 1: الرسم من المعادلة
            print("🎨 المرحلة 1: الرسم من المعادلة...")
            drawing_result = self.drawing_unit.draw_shape_from_equation(shape)
            cycle_result["stages"]["drawing"] = drawing_result

            if not drawing_result["success"]:
                cycle_result["error"] = "فشل في مرحلة الرسم"
                return cycle_result

            drawn_image = drawing_result["result"]

            # المرحلة 2: الاستنباط من الصورة المرسومة
            print("🔍 المرحلة 2: الاستنباط من الصورة...")
            extraction_result = self.extractor_unit.extract_equation_from_image(drawn_image)
            cycle_result["stages"]["extraction"] = extraction_result

            if not extraction_result["success"]:
                cycle_result["error"] = "فشل في مرحلة الاستنباط"
                return cycle_result

            extracted_features = extraction_result["result"]

            # المرحلة 3: التحليل الفيزيائي
            print("🔬 المرحلة 3: التحليل الفيزيائي...")
            physics_analysis = self.physics_expert.analyze_physics_in_drawing_cycle(
                shape, drawn_image, extracted_features
            )
            cycle_result["physics_analysis"] = {
                "physical_accuracy": physics_analysis.physical_accuracy,
                "contradiction_detected": physics_analysis.contradiction_detected,
                "physics_violations": physics_analysis.physics_violations,
                "realism_score": physics_analysis.realism_score,
                "physics_explanation": physics_analysis.physics_explanation,
                "suggested_corrections": physics_analysis.suggested_corrections
            }

            # المرحلة 4: تحليل الخبير/المستكشف
            print("🧠 المرحلة 4: تحليل الخبير/المستكشف...")
            explorer_feedback = self.expert_bridge.analyze_drawing_extraction_cycle(
                shape, drawn_image, extracted_features
            )
            cycle_result["stages"]["expert_analysis"] = {
                "extraction_accuracy": explorer_feedback.extraction_accuracy,
                "drawing_fidelity": explorer_feedback.drawing_fidelity,
                "pattern_recognition": explorer_feedback.pattern_recognition_score,
                "suggestions": explorer_feedback.suggested_improvements
            }

            # المرحلة 5: تقييم التوازن الفني-الفيزيائي
            print("🎨 المرحلة 5: تقييم التوازن الفني-الفيزيائي...")
            artistic_score = (explorer_feedback.drawing_fidelity + explorer_feedback.pattern_recognition_score) / 2
            artistic_physics_balance = self.physics_expert.evaluate_artistic_physics_balance(
                shape, physics_analysis, artistic_score
            )
            cycle_result["artistic_physics_balance"] = {
                "artistic_beauty": artistic_physics_balance.artistic_beauty,
                "physical_accuracy": artistic_physics_balance.physical_accuracy,
                "creative_interpretation": artistic_physics_balance.creative_interpretation,
                "overall_harmony": artistic_physics_balance.overall_harmony,
                "balance_recommendations": artistic_physics_balance.balance_recommendations
            }

            # المرحلة 6: التعلم والتحسين المتكامل
            if learn_from_cycle:
                print("📈 المرحلة 6: التعلم والتحسين المتكامل...")
                improvements = self._apply_integrated_learning_improvements(
                    explorer_feedback, physics_analysis, artistic_physics_balance, shape
                )
                cycle_result["improvements_applied"] = improvements

            # تقييم النجاح الإجمالي (مع الفيزياء)
            overall_score = (
                explorer_feedback.extraction_accuracy * 0.25 +
                explorer_feedback.drawing_fidelity * 0.25 +
                explorer_feedback.pattern_recognition_score * 0.2 +
                physics_analysis.physical_accuracy * 0.2 +
                artistic_physics_balance.overall_harmony * 0.1
            )

            cycle_result["overall_success"] = overall_score > 0.7
            cycle_result["overall_score"] = overall_score

            if cycle_result["overall_success"]:
                self.successful_cycles += 1

            self.integration_cycles += 1
            self.improvement_history.append(overall_score)

            print(f"✅ انتهت الدورة المتكاملة - النتيجة الإجمالية: {overall_score:.2%}")

        except Exception as e:
            cycle_result["error"] = f"خطأ في الدورة المتكاملة: {e}"
            print(f"❌ {cycle_result['error']}")

        return cycle_result

    def _apply_integrated_learning_improvements(self,
                                              feedback: ExplorerFeedback,
                                              physics_analysis: PhysicsAnalysisResult,
                                              balance: ArtisticPhysicsBalance,
                                              shape: ShapeEntity) -> List[str]:
        """تطبيق تحسينات التعلم المتكاملة مع الفيزياء"""
        improvements_applied = []

        # تحسينات الرسم
        if feedback.drawing_fidelity < 0.7:
            improvements_applied.append("تحسين معاملات الرسم")

        # تحسينات الاستنباط
        if feedback.extraction_accuracy < 0.7:
            improvements_applied.append("تحسين خوارزميات الاستنباط")

        # تحسينات التعرف على الأنماط
        if feedback.pattern_recognition_score < 0.7:
            improvements_applied.append("تحسين التعرف على الأنماط")

        # تحسينات فيزيائية جديدة
        if physics_analysis.physical_accuracy < 0.7:
            improvements_applied.append("تحسين الدقة الفيزيائية")

        if physics_analysis.contradiction_detected:
            improvements_applied.append("حل التناقضات الفيزيائية")
            improvements_applied.extend(physics_analysis.suggested_corrections[:2])

        # تحسينات التوازن الفني-الفيزيائي
        if balance.overall_harmony < 0.7:
            improvements_applied.append("تحسين التوازن بين الفن والفيزياء")
            improvements_applied.extend(balance.balance_recommendations[:2])

        # تحسينات خاصة بالفئة
        if shape.category == "حيوانات" and physics_analysis.realism_score < 0.7:
            improvements_applied.append("تحسين واقعية الحيوان")
        elif shape.category == "مباني" and physics_analysis.physical_accuracy < 0.8:
            improvements_applied.append("تحسين الاستقرار الهيكلي")
        elif shape.category == "نباتات" and physics_analysis.realism_score < 0.8:
            improvements_applied.append("تحسين طبيعية النمو")

        return improvements_applied

    def batch_integrated_processing(self, shapes: List[ShapeEntity],
                                  learn_continuously: bool = True) -> Dict[str, Any]:
        """معالجة مجمعة متكاملة لعدة أشكال"""
        print(f"\n🔄 بدء المعالجة المجمعة المتكاملة لـ {len(shapes)} شكل...")

        batch_results = {
            "total_shapes": len(shapes),
            "successful_cycles": 0,
            "failed_cycles": 0,
            "average_score": 0.0,
            "category_performance": {},
            "cycle_results": []
        }

        total_score = 0.0

        for i, shape in enumerate(shapes):
            print(f"\n📊 معالجة الشكل {i+1}/{len(shapes)}: {shape.name}")

            cycle_result = self.execute_integrated_cycle(shape, learn_continuously)
            batch_results["cycle_results"].append(cycle_result)

            if cycle_result["overall_success"]:
                batch_results["successful_cycles"] += 1
                total_score += cycle_result.get("overall_score", 0.0)
            else:
                batch_results["failed_cycles"] += 1

            # تجميع الأداء حسب الفئة
            category = shape.category
            if category not in batch_results["category_performance"]:
                batch_results["category_performance"][category] = {
                    "total": 0,
                    "successful": 0,
                    "scores": []
                }

            batch_results["category_performance"][category]["total"] += 1
            if cycle_result["overall_success"]:
                batch_results["category_performance"][category]["successful"] += 1

            if "overall_score" in cycle_result:
                batch_results["category_performance"][category]["scores"].append(
                    cycle_result["overall_score"]
                )

        # حساب المتوسطات
        if batch_results["successful_cycles"] > 0:
            batch_results["average_score"] = total_score / batch_results["successful_cycles"]

        # حساب متوسط الأداء لكل فئة
        for category, performance in batch_results["category_performance"].items():
            if performance["scores"]:
                performance["average_score"] = np.mean(performance["scores"])
                performance["success_rate"] = performance["successful"] / performance["total"]

        print(f"\n📊 نتائج المعالجة المجمعة:")
        print(f"   ✅ نجح: {batch_results['successful_cycles']}/{batch_results['total_shapes']}")
        print(f"   📈 المتوسط: {batch_results['average_score']:.2%}")

        return batch_results

    def optimize_integration_parameters(self, test_shapes: List[ShapeEntity]) -> Dict[str, Any]:
        """تحسين معاملات التكامل"""
        print("\n🔧 بدء تحسين معاملات التكامل...")

        # تشغيل دورات اختبار
        baseline_results = self.batch_integrated_processing(test_shapes, learn_continuously=False)

        # تطبيق التعلم
        learning_results = self.batch_integrated_processing(test_shapes, learn_continuously=True)

        # مقارنة النتائج
        improvement = learning_results["average_score"] - baseline_results["average_score"]

        optimization_result = {
            "baseline_performance": baseline_results["average_score"],
            "optimized_performance": learning_results["average_score"],
            "improvement": improvement,
            "optimization_successful": improvement > 0.05,
            "expert_recommendations": self._get_optimization_recommendations()
        }

        print(f"📊 نتائج التحسين:")
        print(f"   📈 الأداء الأساسي: {baseline_results['average_score']:.2%}")
        print(f"   🚀 الأداء المحسن: {learning_results['average_score']:.2%}")
        print(f"   ⬆️ التحسن: {improvement:.2%}")

        return optimization_result

    def _get_optimization_recommendations(self) -> List[str]:
        """الحصول على توصيات التحسين"""
        recommendations = []

        if self.integration_cycles > 0:
            success_rate = self.successful_cycles / self.integration_cycles

            if success_rate < 0.6:
                recommendations.append("تحسين خوارزميات الرسم والاستنباط")
                recommendations.append("زيادة دقة معاملات المعادلات")

            if len(self.improvement_history) > 5:
                recent_trend = np.mean(self.improvement_history[-5:]) - np.mean(self.improvement_history[:5])
                if recent_trend > 0:
                    recommendations.append("النظام يتحسن - استمرار التعلم")
                else:
                    recommendations.append("مراجعة استراتيجية التعلم")

        return recommendations

    def generate_integration_report(self) -> Dict[str, Any]:
        """توليد تقرير التكامل"""

        # الحصول على حالة الخبير
        expert_status = self.expert_bridge.get_system_status()

        report = {
            "integration_summary": {
                "total_cycles": self.integration_cycles,
                "successful_cycles": self.successful_cycles,
                "success_rate": self.successful_cycles / self.integration_cycles if self.integration_cycles > 0 else 0,
                "average_improvement": np.mean(self.improvement_history) if self.improvement_history else 0
            },
            "expert_system_status": expert_status,
            "performance_trend": self._analyze_performance_trend(),
            "recommendations": self._generate_integration_recommendations(),
            "system_health": self._assess_system_health()
        }

        return report

    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """تحليل اتجاه الأداء"""
        if len(self.improvement_history) < 3:
            return {"status": "بيانات غير كافية", "trend": "غير محدد"}

        recent_scores = self.improvement_history[-5:]
        early_scores = self.improvement_history[:5]

        trend_direction = np.mean(recent_scores) - np.mean(early_scores)

        if trend_direction > 0.1:
            trend = "تحسن ممتاز"
        elif trend_direction > 0.05:
            trend = "تحسن جيد"
        elif trend_direction > -0.05:
            trend = "مستقر"
        else:
            trend = "يحتاج تحسين"

        return {
            "trend": trend,
            "trend_value": trend_direction,
            "recent_average": np.mean(recent_scores),
            "early_average": np.mean(early_scores)
        }

    def _generate_integration_recommendations(self) -> List[str]:
        """توليد توصيات التكامل"""
        recommendations = []

        if self.integration_cycles == 0:
            recommendations.append("بدء تشغيل دورات التكامل لجمع البيانات")
            return recommendations

        success_rate = self.successful_cycles / self.integration_cycles

        if success_rate > 0.8:
            recommendations.append("الأداء ممتاز - يمكن إضافة ميزات متقدمة")
        elif success_rate > 0.6:
            recommendations.append("الأداء جيد - تحسينات طفيفة مطلوبة")
        else:
            recommendations.append("الأداء يحتاج تحسين كبير")
            recommendations.append("مراجعة خوارزميات الرسم والاستنباط")

        # توصيات بناءً على اتجاه الأداء
        trend = self._analyze_performance_trend()
        if trend["trend"] == "يحتاج تحسين":
            recommendations.append("تطبيق استراتيجيات تعلم جديدة")

        return recommendations

    def _assess_system_health(self) -> str:
        """تقييم صحة النظام"""
        if self.integration_cycles == 0:
            return "غير مختبر"

        success_rate = self.successful_cycles / self.integration_cycles

        if success_rate > 0.8:
            return "ممتاز"
        elif success_rate > 0.6:
            return "جيد"
        elif success_rate > 0.4:
            return "مقبول"
        else:
            return "يحتاج تحسين"

    def save_integration_data(self, filename: str = "integration_data.json"):
        """حفظ بيانات التكامل"""
        try:
            import json
            from datetime import datetime

            integration_data = {
                "integration_cycles": self.integration_cycles,
                "successful_cycles": self.successful_cycles,
                "improvement_history": self.improvement_history,
                "last_updated": datetime.now().isoformat(),
                "system_report": self.generate_integration_report()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(integration_data, f, ensure_ascii=False, indent=2)

            # حفظ معرفة الخبير أيضاً
            self.expert_bridge.save_expert_knowledge()

            print(f"✅ تم حفظ بيانات التكامل في: {filename}")

        except Exception as e:
            print(f"❌ خطأ في حفظ بيانات التكامل: {e}")


def main():
    """اختبار الوحدة المتكاملة"""
    print("🧪 اختبار الوحدة المتكاملة للرسم والاستنباط...")

    # إنشاء الوحدة المتكاملة
    integrated_unit = IntegratedDrawingExtractionUnit()

    # إنشاء أشكال اختبار
    from revolutionary_database import ShapeEntity

    test_shapes = [
        ShapeEntity(
            id=1, name="قطة اختبار", category="حيوانات",
            equation_params={"curve": 0.8, "radius": 0.3},
            geometric_features={"area": 150.0, "perimeter": 45.0, "roundness": 0.7},
            color_properties={"dominant_color": [255, 200, 100]},
            position_info={"center_x": 0.5, "center_y": 0.5, "orientation": 0.0},
            tolerance_thresholds={"euclidean_distance": 0.2},
            created_date="", updated_date=""
        ),
        ShapeEntity(
            id=2, name="بيت اختبار", category="مباني",
            equation_params={"angle": 30.0, "width": 1.0},
            geometric_features={"area": 200.0, "perimeter": 60.0, "roundness": 0.3},
            color_properties={"dominant_color": [150, 100, 80]},
            position_info={"center_x": 0.5, "center_y": 0.4, "orientation": 0.0},
            tolerance_thresholds={"euclidean_distance": 0.25},
            created_date="", updated_date=""
        )
    ]

    # اختبار دورة متكاملة واحدة
    print("\n🔄 اختبار دورة متكاملة واحدة...")
    cycle_result = integrated_unit.execute_integrated_cycle(test_shapes[0])

    # اختبار المعالجة المجمعة
    print("\n🔄 اختبار المعالجة المجمعة...")
    batch_result = integrated_unit.batch_integrated_processing(test_shapes)

    # توليد التقرير
    print("\n📊 توليد تقرير التكامل...")
    report = integrated_unit.generate_integration_report()

    print(f"\n📋 ملخص التقرير:")
    print(f"   🔄 إجمالي الدورات: {report['integration_summary']['total_cycles']}")
    print(f"   ✅ معدل النجاح: {report['integration_summary']['success_rate']:.2%}")
    print(f"   🏥 صحة النظام: {report['system_health']}")

    # حفظ البيانات
    integrated_unit.save_integration_data()


if __name__ == "__main__":
    main()
