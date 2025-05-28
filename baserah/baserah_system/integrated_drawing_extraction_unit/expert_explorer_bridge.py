#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert/Explorer Bridge for Integrated Drawing-Extraction Unit
جسر الخبير/المستكشف للوحدة المتكاملة للرسم والاستنباط

This bridge connects the drawing unit with the extraction unit through the Expert/Explorer system
to enhance accuracy and create a feedback loop for continuous improvement.

هذا الجسر يربط وحدة الرسم بوحدة الاستنباط من خلال نظام الخبير/المستكشف
لتحسين الدقة وإنشاء حلقة تغذية راجعة للتحسين المستمر.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revolutionary_database import ShapeEntity


@dataclass
class ExpertKnowledge:
    """معرفة الخبير"""
    shape_patterns: Dict[str, Any]
    accuracy_metrics: Dict[str, float]
    improvement_suggestions: List[str]
    learned_correlations: Dict[str, Any]


@dataclass
class ExplorerFeedback:
    """تغذية راجعة من المستكشف"""
    extraction_accuracy: float
    drawing_fidelity: float
    pattern_recognition_score: float
    suggested_improvements: List[str]


class ExpertExplorerBridge:
    """جسر الخبير/المستكشف الثوري"""
    
    def __init__(self):
        """تهيئة جسر الخبير/المستكشف"""
        self.expert_knowledge = ExpertKnowledge(
            shape_patterns={},
            accuracy_metrics={},
            improvement_suggestions=[],
            learned_correlations={}
        )
        
        self.exploration_history = []
        self.learning_cycles = 0
        self.accuracy_threshold = 0.85
        
        print("✅ تم تهيئة جسر الخبير/المستكشف الثوري")
    
    def analyze_drawing_extraction_cycle(self, 
                                       original_shape: ShapeEntity,
                                       drawn_image: np.ndarray,
                                       extracted_features: Dict[str, Any]) -> ExplorerFeedback:
        """تحليل دورة الرسم-الاستنباط"""
        print("🔍 تحليل دورة الرسم-الاستنباط...")
        
        # 1. تقييم دقة الاستنباط
        extraction_accuracy = self._evaluate_extraction_accuracy(
            original_shape, extracted_features
        )
        
        # 2. تقييم جودة الرسم
        drawing_fidelity = self._evaluate_drawing_fidelity(
            original_shape, drawn_image
        )
        
        # 3. تقييم التعرف على الأنماط
        pattern_recognition_score = self._evaluate_pattern_recognition(
            original_shape, extracted_features
        )
        
        # 4. اقتراح التحسينات
        suggested_improvements = self._generate_improvement_suggestions(
            extraction_accuracy, drawing_fidelity, pattern_recognition_score
        )
        
        feedback = ExplorerFeedback(
            extraction_accuracy=extraction_accuracy,
            drawing_fidelity=drawing_fidelity,
            pattern_recognition_score=pattern_recognition_score,
            suggested_improvements=suggested_improvements
        )
        
        # 5. تحديث معرفة الخبير
        self._update_expert_knowledge(original_shape, feedback)
        
        print(f"📊 دقة الاستنباط: {extraction_accuracy:.2%}")
        print(f"🎨 جودة الرسم: {drawing_fidelity:.2%}")
        print(f"🧠 التعرف على الأنماط: {pattern_recognition_score:.2%}")
        
        return feedback
    
    def _evaluate_extraction_accuracy(self, 
                                    original_shape: ShapeEntity,
                                    extracted_features: Dict[str, Any]) -> float:
        """تقييم دقة الاستنباط"""
        
        accuracy_scores = []
        
        # مقارنة الخصائص الهندسية
        if "geometric_features" in extracted_features:
            geo_accuracy = self._compare_geometric_features(
                original_shape.geometric_features,
                extracted_features["geometric_features"]
            )
            accuracy_scores.append(geo_accuracy)
        
        # مقارنة الخصائص اللونية
        if "color_properties" in extracted_features:
            color_accuracy = self._compare_color_properties(
                original_shape.color_properties,
                extracted_features["color_properties"]
            )
            accuracy_scores.append(color_accuracy)
        
        # مقارنة معاملات المعادلة
        if "equation_params" in extracted_features:
            equation_accuracy = self._compare_equation_parameters(
                original_shape.equation_params,
                extracted_features["equation_params"]
            )
            accuracy_scores.append(equation_accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def _compare_geometric_features(self, original: Dict[str, float], 
                                  extracted: Dict[str, float]) -> float:
        """مقارنة الخصائص الهندسية"""
        similarities = []
        
        common_features = ["area", "perimeter", "aspect_ratio", "roundness"]
        
        for feature in common_features:
            if feature in original and feature in extracted:
                if original[feature] != 0:
                    similarity = 1.0 - abs(original[feature] - extracted[feature]) / abs(original[feature])
                    similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compare_color_properties(self, original: Dict[str, Any], 
                                extracted: Dict[str, Any]) -> float:
        """مقارنة الخصائص اللونية"""
        if "dominant_color" not in original or "dominant_color" not in extracted:
            return 0.0
        
        orig_color = np.array(original["dominant_color"])
        extr_color = np.array(extracted["dominant_color"])
        
        # حساب المسافة اللونية
        color_distance = np.linalg.norm(orig_color - extr_color)
        max_distance = np.linalg.norm([255, 255, 255])
        
        # تحويل لنسبة تشابه
        similarity = 1.0 - (color_distance / max_distance)
        
        return max(0.0, similarity)
    
    def _compare_equation_parameters(self, original: Dict[str, float], 
                                   extracted: Dict[str, float]) -> float:
        """مقارنة معاملات المعادلة"""
        similarities = []
        
        for param in original:
            if param in extracted:
                if original[param] != 0:
                    similarity = 1.0 - abs(original[param] - extracted[param]) / abs(original[param])
                    similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _evaluate_drawing_fidelity(self, original_shape: ShapeEntity, 
                                 drawn_image: np.ndarray) -> float:
        """تقييم جودة الرسم"""
        
        # تقييم مبسط لجودة الرسم
        fidelity_scores = []
        
        # 1. فحص وجود محتوى في الصورة
        if np.sum(drawn_image) > 0:
            fidelity_scores.append(0.8)
        else:
            fidelity_scores.append(0.0)
        
        # 2. فحص التوزيع اللوني
        if len(drawn_image.shape) == 3:
            color_variance = np.var(drawn_image, axis=(0, 1))
            if np.mean(color_variance) > 100:  # تنوع لوني جيد
                fidelity_scores.append(0.9)
            else:
                fidelity_scores.append(0.6)
        
        # 3. فحص التعقيد المناسب
        edges = self._simple_edge_detection(drawn_image)
        edge_density = np.sum(edges > 0) / (drawn_image.shape[0] * drawn_image.shape[1])
        
        if 0.1 <= edge_density <= 0.4:  # كثافة حواف مناسبة
            fidelity_scores.append(0.85)
        else:
            fidelity_scores.append(0.5)
        
        return np.mean(fidelity_scores)
    
    def _evaluate_pattern_recognition(self, original_shape: ShapeEntity,
                                    extracted_features: Dict[str, Any]) -> float:
        """تقييم التعرف على الأنماط"""
        
        pattern_scores = []
        
        # 1. تطابق الفئة المتوقعة
        if original_shape.category == "حيوانات":
            expected_patterns = ["curved", "organic", "asymmetric"]
        elif original_shape.category == "مباني":
            expected_patterns = ["angular", "geometric", "symmetric"]
        elif original_shape.category == "نباتات":
            expected_patterns = ["organic", "branched", "natural"]
        else:
            expected_patterns = ["general"]
        
        # 2. فحص الأنماط في الخصائص المستنبطة
        if "geometric_features" in extracted_features:
            geo_features = extracted_features["geometric_features"]
            
            # فحص الاستدارة للحيوانات
            if "حيوانات" in original_shape.category and "roundness" in geo_features:
                if geo_features["roundness"] > 0.5:
                    pattern_scores.append(0.8)
                else:
                    pattern_scores.append(0.4)
            
            # فحص الهندسة للمباني
            if "مباني" in original_shape.category and "aspect_ratio" in geo_features:
                if 0.8 <= geo_features["aspect_ratio"] <= 2.0:
                    pattern_scores.append(0.9)
                else:
                    pattern_scores.append(0.5)
        
        return np.mean(pattern_scores) if pattern_scores else 0.5
    
    def _generate_improvement_suggestions(self, extraction_accuracy: float,
                                        drawing_fidelity: float,
                                        pattern_recognition: float) -> List[str]:
        """توليد اقتراحات التحسين"""
        suggestions = []
        
        if extraction_accuracy < 0.7:
            suggestions.append("تحسين خوارزميات استنباط الخصائص الهندسية")
            suggestions.append("تطوير تحليل الألوان والأنسجة")
        
        if drawing_fidelity < 0.7:
            suggestions.append("تحسين دقة الرسم من المعادلات")
            suggestions.append("إضافة تفاصيل أكثر للأشكال المرسومة")
        
        if pattern_recognition < 0.7:
            suggestions.append("تطوير التعرف على أنماط الفئات")
            suggestions.append("تحسين ربط الخصائص بالسياق")
        
        # اقتراحات عامة للتحسين
        if extraction_accuracy > 0.8 and drawing_fidelity > 0.8:
            suggestions.append("النظام يعمل بكفاءة عالية - يمكن إضافة ميزات متقدمة")
        
        return suggestions
    
    def _update_expert_knowledge(self, shape: ShapeEntity, feedback: ExplorerFeedback):
        """تحديث معرفة الخبير"""
        
        # تحديث أنماط الأشكال
        category = shape.category
        if category not in self.expert_knowledge.shape_patterns:
            self.expert_knowledge.shape_patterns[category] = {
                "successful_extractions": 0,
                "total_attempts": 0,
                "common_features": {},
                "accuracy_history": []
            }
        
        pattern = self.expert_knowledge.shape_patterns[category]
        pattern["total_attempts"] += 1
        pattern["accuracy_history"].append(feedback.extraction_accuracy)
        
        if feedback.extraction_accuracy > self.accuracy_threshold:
            pattern["successful_extractions"] += 1
        
        # تحديث مقاييس الدقة
        self.expert_knowledge.accuracy_metrics[category] = {
            "avg_extraction_accuracy": np.mean(pattern["accuracy_history"]),
            "success_rate": pattern["successful_extractions"] / pattern["total_attempts"],
            "last_feedback": feedback.extraction_accuracy
        }
        
        # إضافة اقتراحات التحسين
        self.expert_knowledge.improvement_suggestions.extend(
            feedback.suggested_improvements
        )
        
        # الاحتفاظ بآخر 10 اقتراحات فقط
        self.expert_knowledge.improvement_suggestions = \
            self.expert_knowledge.improvement_suggestions[-10:]
        
        self.learning_cycles += 1
    
    def get_expert_recommendations(self, shape_category: str) -> Dict[str, Any]:
        """الحصول على توصيات الخبير"""
        
        if shape_category in self.expert_knowledge.shape_patterns:
            pattern = self.expert_knowledge.shape_patterns[shape_category]
            metrics = self.expert_knowledge.accuracy_metrics.get(shape_category, {})
            
            return {
                "category": shape_category,
                "success_rate": metrics.get("success_rate", 0.0),
                "avg_accuracy": metrics.get("avg_extraction_accuracy", 0.0),
                "total_attempts": pattern["total_attempts"],
                "recommendations": self._generate_category_recommendations(shape_category),
                "confidence_level": self._calculate_confidence_level(shape_category)
            }
        
        return {
            "category": shape_category,
            "status": "غير معروف - يحتاج المزيد من البيانات",
            "recommendations": ["جمع المزيد من العينات لهذه الفئة"]
        }
    
    def _generate_category_recommendations(self, category: str) -> List[str]:
        """توليد توصيات خاصة بالفئة"""
        
        if category not in self.expert_knowledge.accuracy_metrics:
            return ["جمع المزيد من البيانات لهذه الفئة"]
        
        metrics = self.expert_knowledge.accuracy_metrics[category]
        recommendations = []
        
        if metrics["success_rate"] < 0.6:
            recommendations.append(f"تحسين خوارزميات التعرف على {category}")
            recommendations.append(f"إضافة المزيد من العينات لفئة {category}")
        
        if metrics["avg_extraction_accuracy"] < 0.7:
            recommendations.append(f"تطوير استنباط الخصائص المميزة لـ {category}")
        
        if metrics["success_rate"] > 0.8:
            recommendations.append(f"الأداء ممتاز لفئة {category} - يمكن استخدامها كمرجع")
        
        return recommendations
    
    def _calculate_confidence_level(self, category: str) -> str:
        """حساب مستوى الثقة"""
        
        if category not in self.expert_knowledge.accuracy_metrics:
            return "منخفض"
        
        metrics = self.expert_knowledge.accuracy_metrics[category]
        pattern = self.expert_knowledge.shape_patterns[category]
        
        # حساب الثقة بناءً على عدد المحاولات والدقة
        attempts = pattern["total_attempts"]
        accuracy = metrics["avg_extraction_accuracy"]
        
        if attempts >= 10 and accuracy > 0.8:
            return "عالي"
        elif attempts >= 5 and accuracy > 0.6:
            return "متوسط"
        else:
            return "منخفض"
    
    def _simple_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """كشف الحواف المبسط"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # مرشح Sobel مبسط
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        edges_x = self._convolve2d(gray, kernel_x)
        edges_y = self._convolve2d(gray, kernel_y)
        
        return np.sqrt(edges_x**2 + edges_y**2)
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """تطبيق مرشح ثنائي الأبعاد"""
        result = np.zeros_like(image, dtype=np.float32)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)
        
        return result
    
    def save_expert_knowledge(self, filename: str = "expert_knowledge.json"):
        """حفظ معرفة الخبير"""
        try:
            knowledge_data = {
                "shape_patterns": self.expert_knowledge.shape_patterns,
                "accuracy_metrics": self.expert_knowledge.accuracy_metrics,
                "improvement_suggestions": self.expert_knowledge.improvement_suggestions,
                "learning_cycles": self.learning_cycles,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ تم حفظ معرفة الخبير في: {filename}")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ معرفة الخبير: {e}")
    
    def load_expert_knowledge(self, filename: str = "expert_knowledge.json"):
        """تحميل معرفة الخبير"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            self.expert_knowledge.shape_patterns = knowledge_data.get("shape_patterns", {})
            self.expert_knowledge.accuracy_metrics = knowledge_data.get("accuracy_metrics", {})
            self.expert_knowledge.improvement_suggestions = knowledge_data.get("improvement_suggestions", [])
            self.learning_cycles = knowledge_data.get("learning_cycles", 0)
            
            print(f"✅ تم تحميل معرفة الخبير من: {filename}")
            
        except FileNotFoundError:
            print(f"⚠️ ملف معرفة الخبير غير موجود: {filename}")
        except Exception as e:
            print(f"❌ خطأ في تحميل معرفة الخبير: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """حالة نظام الخبير/المستكشف"""
        return {
            "learning_cycles": self.learning_cycles,
            "categories_learned": len(self.expert_knowledge.shape_patterns),
            "total_improvements": len(self.expert_knowledge.improvement_suggestions),
            "accuracy_threshold": self.accuracy_threshold,
            "expert_confidence": self._calculate_overall_confidence()
        }
    
    def _calculate_overall_confidence(self) -> str:
        """حساب الثقة الإجمالية"""
        if not self.expert_knowledge.accuracy_metrics:
            return "منخفض"
        
        avg_accuracy = np.mean([
            metrics["avg_extraction_accuracy"] 
            for metrics in self.expert_knowledge.accuracy_metrics.values()
        ])
        
        if avg_accuracy > 0.8:
            return "عالي"
        elif avg_accuracy > 0.6:
            return "متوسط"
        else:
            return "منخفض"


def main():
    """اختبار جسر الخبير/المستكشف"""
    print("🧪 اختبار جسر الخبير/المستكشف...")
    
    # إنشاء الجسر
    bridge = ExpertExplorerBridge()
    
    # إنشاء شكل اختبار
    from revolutionary_database import ShapeEntity
    
    test_shape = ShapeEntity(
        id=1,
        name="قطة اختبار",
        category="حيوانات",
        equation_params={"curve": 0.8, "radius": 0.3},
        geometric_features={"area": 150.0, "roundness": 0.7},
        color_properties={"dominant_color": [255, 200, 100]},
        position_info={"center_x": 0.5, "center_y": 0.5},
        tolerance_thresholds={},
        created_date="",
        updated_date=""
    )
    
    # إنشاء صورة وخصائص مستنبطة للاختبار
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    extracted_features = {
        "geometric_features": {"area": 145.0, "roundness": 0.65},
        "color_properties": {"dominant_color": [250, 195, 105]}
    }
    
    # تحليل الدورة
    feedback = bridge.analyze_drawing_extraction_cycle(
        test_shape, test_image, extracted_features
    )
    
    # عرض التوصيات
    recommendations = bridge.get_expert_recommendations("حيوانات")
    print(f"\n📋 توصيات الخبير:")
    for rec in recommendations.get("recommendations", []):
        print(f"   • {rec}")
    
    # عرض حالة النظام
    status = bridge.get_system_status()
    print(f"\n📊 حالة النظام:")
    print(f"   🔄 دورات التعلم: {status['learning_cycles']}")
    print(f"   📂 الفئات المتعلمة: {status['categories_learned']}")
    print(f"   🎯 مستوى الثقة: {status['expert_confidence']}")


if __name__ == "__main__":
    main()
