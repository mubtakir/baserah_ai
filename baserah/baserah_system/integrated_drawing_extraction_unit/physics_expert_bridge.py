#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Expert Bridge for Integrated Drawing-Extraction Unit
جسر الخبير الفيزيائي للوحدة المتكاملة للرسم والاستنباط

This bridge integrates the Physical Thinking Unit with the Expert/Explorer system
to ensure physical accuracy in drawing and extraction processes.

هذا الجسر يدمج وحدة التفكير الفيزيائي مع نظام الخبير/المستكشف
لضمان الدقة الفيزيائية في عمليات الرسم والاستنباط.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revolutionary_database import ShapeEntity

# استيراد محلي للاختبار
try:
    from .expert_explorer_bridge import ExpertExplorerBridge, ExplorerFeedback
except ImportError:
    from expert_explorer_bridge import ExpertExplorerBridge, ExplorerFeedback

# استيراد وحدة التفكير الفيزيائي
try:
    from physical_thinking.revolutionary_physics_engine import RevolutionaryPhysicsEngine
    from physical_thinking.advanced_contradiction_detector import AdvancedContradictionDetector
    PHYSICS_UNIT_AVAILABLE = True
    print("✅ وحدة التفكير الفيزيائي متاحة")
except ImportError as e:
    PHYSICS_UNIT_AVAILABLE = False
    print(f"⚠️ وحدة التفكير الفيزيائي غير متاحة: {e}")


@dataclass
class PhysicsAnalysisResult:
    """نتيجة التحليل الفيزيائي"""
    physical_accuracy: float
    contradiction_detected: bool
    physics_violations: List[str]
    suggested_corrections: List[str]
    realism_score: float
    physics_explanation: str


@dataclass
class ArtisticPhysicsBalance:
    """توازن الفن والفيزياء"""
    artistic_beauty: float
    physical_accuracy: float
    creative_interpretation: float
    overall_harmony: float
    balance_recommendations: List[str]


class PhysicsExpertBridge:
    """جسر الخبير الفيزيائي الثوري"""

    def __init__(self, expert_bridge: ExpertExplorerBridge):
        """تهيئة جسر الخبير الفيزيائي"""
        self.expert_bridge = expert_bridge
        self.physics_available = PHYSICS_UNIT_AVAILABLE

        if self.physics_available:
            try:
                self.physics_engine = RevolutionaryPhysicsEngine()
                self.contradiction_detector = AdvancedContradictionDetector()
                print("✅ تم تهيئة جسر الخبير الفيزيائي")
            except Exception as e:
                print(f"⚠️ خطأ في تهيئة المحرك الفيزيائي: {e}")
                self.physics_available = False

        # إحصائيات التحليل الفيزيائي
        self.physics_analyses = 0
        self.contradictions_found = 0
        self.corrections_applied = 0

        if not self.physics_available:
            print("⚠️ سيتم استخدام تحليل فيزيائي مبسط")

    def analyze_physics_in_drawing_cycle(self,
                                       original_shape: ShapeEntity,
                                       drawn_image: np.ndarray,
                                       extracted_features: Dict[str, Any]) -> PhysicsAnalysisResult:
        """تحليل الفيزياء في دورة الرسم-الاستنباط"""
        print("🔬 بدء التحليل الفيزيائي للدورة...")

        if self.physics_available:
            return self._advanced_physics_analysis(original_shape, drawn_image, extracted_features)
        else:
            return self._simple_physics_analysis(original_shape, drawn_image, extracted_features)

    def _advanced_physics_analysis(self, shape: ShapeEntity, image: np.ndarray,
                                 features: Dict[str, Any]) -> PhysicsAnalysisResult:
        """التحليل الفيزيائي المتقدم"""

        # 1. تحليل الدقة الفيزيائية
        physical_accuracy = self._evaluate_physical_accuracy(shape, features)

        # 2. كشف التناقضات الفيزيائية
        contradictions = self.contradiction_detector.detect_contradictions({
            "shape_data": shape,
            "image_data": image,
            "extracted_features": features
        })

        contradiction_detected = len(contradictions) > 0
        if contradiction_detected:
            self.contradictions_found += 1

        # 3. تقييم الواقعية
        realism_score = self._calculate_realism_score(shape, features)

        # 4. توليد التصحيحات المقترحة
        suggested_corrections = self._generate_physics_corrections(
            shape, contradictions, physical_accuracy
        )

        # 5. توليد الشرح الفيزيائي
        physics_explanation = self._generate_physics_explanation(
            shape, contradictions, physical_accuracy
        )

        self.physics_analyses += 1

        return PhysicsAnalysisResult(
            physical_accuracy=physical_accuracy,
            contradiction_detected=contradiction_detected,
            physics_violations=[c.get("description", "") for c in contradictions],
            suggested_corrections=suggested_corrections,
            realism_score=realism_score,
            physics_explanation=physics_explanation
        )

    def _simple_physics_analysis(self, shape: ShapeEntity, image: np.ndarray,
                                features: Dict[str, Any]) -> PhysicsAnalysisResult:
        """التحليل الفيزيائي المبسط"""

        # تحليل مبسط للفيزياء
        physical_accuracy = self._simple_physics_check(shape, features)

        # فحص التناقضات الأساسية
        contradictions = self._simple_contradiction_check(shape, features)

        # حساب الواقعية المبسط
        realism_score = min(physical_accuracy + 0.2, 1.0)

        suggested_corrections = []
        if physical_accuracy < 0.7:
            suggested_corrections.append("تحسين الدقة الفيزيائية للشكل")

        if contradictions:
            suggested_corrections.append("حل التناقضات الفيزيائية المكتشفة")

        physics_explanation = f"تحليل فيزيائي مبسط: دقة {physical_accuracy:.2%}"

        return PhysicsAnalysisResult(
            physical_accuracy=physical_accuracy,
            contradiction_detected=len(contradictions) > 0,
            physics_violations=contradictions,
            suggested_corrections=suggested_corrections,
            realism_score=realism_score,
            physics_explanation=physics_explanation
        )

    def _evaluate_physical_accuracy(self, shape: ShapeEntity,
                                   features: Dict[str, Any]) -> float:
        """تقييم الدقة الفيزيائية"""
        accuracy_factors = []

        # 1. فحص الجاذبية والتوازن
        if shape.category == "حيوانات":
            # فحص وضعية الحيوان
            if "نائمة" in shape.name:
                accuracy_factors.append(0.9)  # وضعية مستقرة
            elif "واقفة" in shape.name:
                accuracy_factors.append(0.8)  # تحتاج توازن
            elif "تقفز" in shape.name:
                accuracy_factors.append(0.6)  # تحتاج فحص مسار

        elif shape.category == "مباني":
            # فحص الاستقرار الهيكلي
            if "geometric_features" in features:
                aspect_ratio = features["geometric_features"].get("aspect_ratio", 1.0)
                if 0.5 <= aspect_ratio <= 2.0:
                    accuracy_factors.append(0.9)  # نسبة مستقرة
                else:
                    accuracy_factors.append(0.5)  # قد تكون غير مستقرة

        elif shape.category == "نباتات":
            # فحص النمو الطبيعي
            if "كبيرة" in shape.name:
                accuracy_factors.append(0.8)  # نمو طبيعي
            else:
                accuracy_factors.append(0.9)

        # 2. فحص الخصائص الفيزيائية
        if "geometric_features" in features:
            geo_features = features["geometric_features"]

            # فحص الكثافة المنطقية
            area = geo_features.get("area", 0)
            perimeter = geo_features.get("perimeter", 0)

            if area > 0 and perimeter > 0:
                # نسبة المساحة للمحيط يجب أن تكون منطقية
                ratio = area / (perimeter ** 2)
                if 0.01 <= ratio <= 0.1:
                    accuracy_factors.append(0.9)
                else:
                    accuracy_factors.append(0.6)

        return np.mean(accuracy_factors) if accuracy_factors else 0.5

    def _simple_physics_check(self, shape: ShapeEntity, features: Dict[str, Any]) -> float:
        """فحص فيزيائي مبسط"""
        score = 0.7  # نقطة بداية

        # فحص الخصائص الأساسية
        if "geometric_features" in features:
            geo = features["geometric_features"]

            # فحص المنطقية
            if geo.get("area", 0) > 0:
                score += 0.1

            if 0.1 <= geo.get("roundness", 0) <= 1.0:
                score += 0.1

            if geo.get("aspect_ratio", 0) > 0:
                score += 0.1

        return min(score, 1.0)

    def _simple_contradiction_check(self, shape: ShapeEntity,
                                   features: Dict[str, Any]) -> List[str]:
        """فحص التناقضات المبسط"""
        contradictions = []

        # فحص التناقضات الأساسية
        if "geometric_features" in features:
            geo = features["geometric_features"]

            # مساحة سالبة
            if geo.get("area", 0) <= 0:
                contradictions.append("مساحة غير منطقية")

            # نسبة عرض/ارتفاع غير منطقية
            aspect_ratio = geo.get("aspect_ratio", 1.0)
            if aspect_ratio <= 0 or aspect_ratio > 10:
                contradictions.append("نسبة أبعاد غير منطقية")

        return contradictions

    def _calculate_realism_score(self, shape: ShapeEntity,
                               features: Dict[str, Any]) -> float:
        """حساب نتيجة الواقعية"""
        realism_factors = []

        # عوامل الواقعية حسب الفئة
        if shape.category == "حيوانات":
            # واقعية الحيوانات
            if "color_properties" in features:
                color = features["color_properties"].get("dominant_color", [128, 128, 128])
                # ألوان طبيعية للحيوانات
                if any(c > 200 for c in color) or any(c < 50 for c in color):
                    realism_factors.append(0.8)  # ألوان طبيعية
                else:
                    realism_factors.append(0.6)

        elif shape.category == "مباني":
            # واقعية المباني
            if "geometric_features" in features:
                roundness = features["geometric_features"].get("roundness", 0.5)
                if roundness < 0.5:  # مباني عادة أكثر زاوية
                    realism_factors.append(0.9)
                else:
                    realism_factors.append(0.7)

        # إضافة عوامل أخرى
        realism_factors.append(0.8)  # عامل أساسي

        return np.mean(realism_factors)

    def _generate_physics_corrections(self, shape: ShapeEntity,
                                    contradictions: List[Dict],
                                    accuracy: float) -> List[str]:
        """توليد التصحيحات الفيزيائية"""
        corrections = []

        # تصحيحات بناءً على التناقضات
        for contradiction in contradictions:
            if "gravity" in contradiction.get("type", ""):
                corrections.append("تطبيق قوانين الجاذبية على الشكل")
            elif "balance" in contradiction.get("type", ""):
                corrections.append("تحسين توازن الشكل")
            elif "structure" in contradiction.get("type", ""):
                corrections.append("تقوية البنية الهيكلية")

        # تصحيحات بناءً على الدقة
        if accuracy < 0.6:
            corrections.append("مراجعة شاملة للخصائص الفيزيائية")
        elif accuracy < 0.8:
            corrections.append("تحسينات طفيفة في الدقة الفيزيائية")

        # تصحيحات خاصة بالفئة
        if shape.category == "حيوانات":
            if "تقفز" in shape.name:
                corrections.append("تصحيح مسار القفز وفقاً لقوانين الحركة")
        elif shape.category == "مباني":
            corrections.append("فحص الاستقرار الهيكلي")

        return corrections

    def _generate_physics_explanation(self, shape: ShapeEntity,
                                    contradictions: List[Dict],
                                    accuracy: float) -> str:
        """توليد الشرح الفيزيائي"""

        explanation = f"التحليل الفيزيائي لـ {shape.name}:\n"
        explanation += f"الدقة الفيزيائية: {accuracy:.2%}\n"

        if contradictions:
            explanation += f"تناقضات مكتشفة: {len(contradictions)}\n"
            for i, contradiction in enumerate(contradictions[:3], 1):
                explanation += f"{i}. {contradiction.get('description', 'تناقض غير محدد')}\n"
        else:
            explanation += "لا توجد تناقضات فيزيائية واضحة\n"

        # إضافة نصائح فيزيائية
        if shape.category == "حيوانات":
            explanation += "نصيحة: تأكد من واقعية الوضعية والحركة"
        elif shape.category == "مباني":
            explanation += "نصيحة: راجع الاستقرار الهيكلي والنسب"
        elif shape.category == "نباتات":
            explanation += "نصيحة: تحقق من طبيعية النمو والتفرع"

        return explanation

    def evaluate_artistic_physics_balance(self, shape: ShapeEntity,
                                        physics_result: PhysicsAnalysisResult,
                                        artistic_score: float) -> ArtisticPhysicsBalance:
        """تقييم توازن الفن والفيزياء"""

        # حساب الجمال الفني
        artistic_beauty = artistic_score

        # الدقة الفيزيائية
        physical_accuracy = physics_result.physical_accuracy

        # التفسير الإبداعي (السماح ببعض الحرية الفنية)
        creative_interpretation = self._calculate_creative_freedom(shape, physics_result)

        # التناغم الإجمالي
        overall_harmony = (
            artistic_beauty * 0.4 +
            physical_accuracy * 0.4 +
            creative_interpretation * 0.2
        )

        # توصيات التوازن
        balance_recommendations = self._generate_balance_recommendations(
            artistic_beauty, physical_accuracy, creative_interpretation
        )

        return ArtisticPhysicsBalance(
            artistic_beauty=artistic_beauty,
            physical_accuracy=physical_accuracy,
            creative_interpretation=creative_interpretation,
            overall_harmony=overall_harmony,
            balance_recommendations=balance_recommendations
        )

    def _calculate_creative_freedom(self, shape: ShapeEntity,
                                  physics_result: PhysicsAnalysisResult) -> float:
        """حساب الحرية الإبداعية المسموحة"""

        base_freedom = 0.7  # حرية أساسية

        # تقليل الحرية مع زيادة التناقضات الخطيرة
        serious_violations = [v for v in physics_result.physics_violations
                            if "خطير" in v or "مستحيل" in v]

        freedom_penalty = len(serious_violations) * 0.2

        # زيادة الحرية للأعمال الفنية
        if "فني" in shape.name or "إبداعي" in shape.name:
            base_freedom += 0.2

        return max(0.1, min(1.0, base_freedom - freedom_penalty))

    def _generate_balance_recommendations(self, artistic: float,
                                        physical: float,
                                        creative: float) -> List[str]:
        """توليد توصيات التوازن"""
        recommendations = []

        if physical < 0.6:
            recommendations.append("تحسين الدقة الفيزيائية مع الحفاظ على الجمال الفني")

        if artistic < 0.6:
            recommendations.append("إضافة لمسات فنية مع احترام القوانين الفيزيائية")

        if abs(artistic - physical) > 0.4:
            recommendations.append("تحقيق توازن أفضل بين الفن والفيزياء")

        if creative > 0.8:
            recommendations.append("استغلال الحرية الإبداعية بشكل أكثر جرأة")

        return recommendations

    def get_physics_expert_status(self) -> Dict[str, Any]:
        """حالة خبير الفيزياء"""
        return {
            "physics_unit_available": self.physics_available,
            "total_analyses": self.physics_analyses,
            "contradictions_found": self.contradictions_found,
            "corrections_applied": self.corrections_applied,
            "accuracy_rate": (self.physics_analyses - self.contradictions_found) / max(1, self.physics_analyses),
            "expert_level": self._determine_expert_level()
        }

    def _determine_expert_level(self) -> str:
        """تحديد مستوى الخبرة"""
        if self.physics_analyses < 5:
            return "مبتدئ"
        elif self.physics_analyses < 20:
            return "متوسط"
        elif self.physics_analyses < 50:
            return "متقدم"
        else:
            return "خبير"


def main():
    """اختبار جسر الخبير الفيزيائي"""
    print("🧪 اختبار جسر الخبير الفيزيائي...")

    # إنشاء جسر الخبير العادي
    from expert_explorer_bridge import ExpertExplorerBridge
    expert_bridge = ExpertExplorerBridge()

    # إنشاء جسر الخبير الفيزيائي
    physics_bridge = PhysicsExpertBridge(expert_bridge)

    # إنشاء شكل اختبار
    from revolutionary_database import ShapeEntity

    test_shape = ShapeEntity(
        id=1, name="قطة تقفز", category="حيوانات",
        equation_params={"jump_height": 1.5, "velocity": 3.0},
        geometric_features={"area": 150.0, "aspect_ratio": 1.2},
        color_properties={"dominant_color": [200, 150, 100]},
        position_info={"center_x": 0.5, "center_y": 0.3},
        tolerance_thresholds={}, created_date="", updated_date=""
    )

    # إنشاء صورة وخصائص للاختبار
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    test_features = {
        "geometric_features": {"area": 145.0, "aspect_ratio": 1.1},
        "color_properties": {"dominant_color": [195, 145, 105]}
    }

    # تحليل فيزيائي
    physics_result = physics_bridge.analyze_physics_in_drawing_cycle(
        test_shape, test_image, test_features
    )

    print(f"\n🔬 نتائج التحليل الفيزيائي:")
    print(f"   📊 الدقة الفيزيائية: {physics_result.physical_accuracy:.2%}")
    print(f"   ⚠️ تناقضات مكتشفة: {physics_result.contradiction_detected}")
    print(f"   🎯 نتيجة الواقعية: {physics_result.realism_score:.2%}")
    print(f"   📝 الشرح: {physics_result.physics_explanation}")

    # تقييم التوازن الفني-الفيزيائي
    balance = physics_bridge.evaluate_artistic_physics_balance(
        test_shape, physics_result, 0.8
    )

    print(f"\n🎨 توازن الفن والفيزياء:")
    print(f"   🎨 الجمال الفني: {balance.artistic_beauty:.2%}")
    print(f"   🔬 الدقة الفيزيائية: {balance.physical_accuracy:.2%}")
    print(f"   💡 الحرية الإبداعية: {balance.creative_interpretation:.2%}")
    print(f"   🌟 التناغم الإجمالي: {balance.overall_harmony:.2%}")


if __name__ == "__main__":
    main()
