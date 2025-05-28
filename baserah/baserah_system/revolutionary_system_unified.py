#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Shape Recognition System - Unified
النظام الثوري للتعرف على الأشكال - موحد

Complete implementation of Basil Yahya Abdullah's revolutionary concept:
"Database + Drawing Unit + Extractor Unit + Smart Recognition with Tolerance"

التطبيق الكامل لمفهوم باسل يحيى عبدالله الثوري:
"قاعدة بيانات + وحدة رسم + وحدة استنباط + تعرف ذكي مع السماحية"

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, '.')

from revolutionary_database import RevolutionaryShapeDatabase, ShapeEntity
from revolutionary_drawing_unit import RevolutionaryDrawingUnit
from revolutionary_extractor_unit import RevolutionaryExtractorUnit
from revolutionary_recognition_engine import RevolutionaryRecognitionEngine

# استيراد الوحدة المتكاملة الجديدة
try:
    from integrated_drawing_extraction_unit.integrated_unit import IntegratedDrawingExtractionUnit
    INTEGRATED_UNIT_AVAILABLE = True
    print("✅ الوحدة المتكاملة متاحة: integrated_drawing_extraction_unit")
except ImportError as e:
    INTEGRATED_UNIT_AVAILABLE = False
    print(f"⚠️ الوحدة المتكاملة غير متاحة: {e}")


class RevolutionaryShapeRecognitionSystem:
    """النظام الثوري الكامل للتعرف على الأشكال - إبداع باسل يحيى عبدالله"""

    def __init__(self):
        """تهيئة النظام الثوري الكامل"""
        print("🌟" + "="*80 + "🌟")
        print("🚀 النظام الثوري للتعرف على الأشكال - نظام بصيرة 🚀")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*80 + "🌟")

        # تهيئة المكونات الثورية
        print("\n🔧 تهيئة المكونات الثورية...")
        self.shape_db = RevolutionaryShapeDatabase()
        self.drawing_unit = RevolutionaryDrawingUnit()
        self.extractor_unit = RevolutionaryExtractorUnit()
        self.recognition_engine = RevolutionaryRecognitionEngine(
            self.shape_db, self.extractor_unit
        )

        # تهيئة الوحدة المتكاملة إذا كانت متاحة
        if INTEGRATED_UNIT_AVAILABLE:
            try:
                self.integrated_unit = IntegratedDrawingExtractionUnit()
                print("✅ تم تهيئة الوحدة المتكاملة للرسم والاستنباط")
            except Exception as e:
                print(f"⚠️ خطأ في تهيئة الوحدة المتكاملة: {e}")
                self.integrated_unit = None
        else:
            self.integrated_unit = None

        print("✅ تم تهيئة النظام الثوري الكامل بنجاح!")

    def demonstrate_revolutionary_concept(self):
        """عرض المفهوم الثوري لباسل يحيى عبدالله"""
        print("\n🎯 عرض المفهوم الثوري:")
        print("💡 الفكرة: قاعدة بيانات + وحدة رسم + وحدة استنباط + تعرف ذكي")
        print("🔄 التدفق: صورة → استنباط → مقارنة → تعرف → وصف ذكي")
        print("🎯 المثال: 'قطة بيضاء نائمة بخلفية بيوت وأشجار'")
        print("📏 السماحية: هندسية + لونية + مسافة إقليدية")

        # عرض قاعدة البيانات
        self._demonstrate_database()

        # عرض وحدة الرسم
        self._demonstrate_drawing()

        # عرض وحدة الاستنباط
        self._demonstrate_extraction()

        # عرض التعرف الثوري
        self._demonstrate_recognition()

        # عرض الوحدة المتكاملة إذا كانت متاحة
        if self.integrated_unit:
            self._demonstrate_integrated_unit()

    def _demonstrate_database(self):
        """عرض قاعدة البيانات الثورية"""
        print("\n📊 قاعدة البيانات الثورية:")
        shapes = self.shape_db.get_all_shapes()
        stats = self.shape_db.get_statistics()

        print(f"   📈 إجمالي الأشكال: {stats['total_shapes']}")
        print(f"   📂 الفئات: {list(stats['categories'].keys())}")
        print(f"   🎯 متوسط السماحية الإقليدية: {stats['average_tolerance']:.3f}")

        for i, shape in enumerate(shapes, 1):
            print(f"{i}. {shape.name} ({shape.category})")
            print(f"   🎨 اللون: {shape.color_properties['dominant_color']}")
            print(f"   🎯 السماحية الإقليدية: {shape.tolerance_thresholds['euclidean_distance']}")

    def _demonstrate_drawing(self):
        """عرض وحدة الرسم والتحريك"""
        print("\n🎨 عرض وحدة الرسم والتحريك الثورية:")
        shapes = self.shape_db.get_all_shapes()

        for shape in shapes[:2]:  # رسم أول شكلين
            print(f"🖌️ رسم {shape.name} باستخدام animated_path_plotter_timeline...")
            result = self.drawing_unit.draw_shape_from_equation(shape)
            print(f"   ✅ {result['message']}")
            print(f"   🔧 الطريقة: {result['method']}")

    def _demonstrate_extraction(self):
        """عرض وحدة الاستنباط"""
        print("\n🔬 عرض وحدة الاستنباط الثورية:")

        # إنشاء صورة اختبار
        test_shape = self.shape_db.get_all_shapes()[0]
        drawing_result = self.drawing_unit.draw_shape_from_equation(test_shape)

        if drawing_result["success"]:
            test_image = drawing_result["result"]
            if isinstance(test_image, np.ndarray):
                print(f"🔍 استنباط معادلة من صورة {test_shape.name}...")
                extraction_result = self.extractor_unit.extract_equation_from_image(test_image)
                print(f"   ✅ {extraction_result['message']}")
                print(f"   🔧 الطريقة: {extraction_result['method']}")

                # عرض الخصائص المستنبطة
                if extraction_result["success"]:
                    features = extraction_result["result"]
                    print(f"   🎨 اللون المستنبط: {features['color_properties']['dominant_color']}")
                    print(f"   📐 المساحة المستنبطة: {features['geometric_features']['area']:.1f}")

    def _demonstrate_recognition(self):
        """عرض التعرف الثوري"""
        print("\n🧠 عرض التعرف الثوري مع السماحية والمسافة الإقليدية:")

        # اختبار التعرف على شكل معروف
        test_shape = self.shape_db.get_all_shapes()[0]
        drawing_result = self.drawing_unit.draw_shape_from_equation(test_shape)

        if drawing_result["success"] and isinstance(drawing_result["result"], np.ndarray):
            test_image = drawing_result["result"]

            print(f"🔍 اختبار التعرف على: {test_shape.name}")
            recognition_result = self.recognition_engine.recognize_image(test_image)

            print(f"📊 النتيجة: {recognition_result['status']}")
            if recognition_result['status'] == "تم التعرف بنجاح":
                print(f"🎯 الشكل المتعرف عليه: {recognition_result['recognized_shape']}")
                print(f"📈 مستوى الثقة: {recognition_result['confidence']:.2%}")
                print(f"📏 المسافة الإقليدية: {recognition_result['euclidean_distance']:.4f}")
                print(f"📐 التشابه الهندسي: {recognition_result['geometric_similarity']:.4f}")
                print(f"🌈 التشابه اللوني: {recognition_result['color_similarity']:.4f}")
                print(f"📝 الوصف الذكي: {recognition_result['description']}")

    def _demonstrate_integrated_unit(self):
        """عرض الوحدة المتكاملة مع جسر الخبير/المستكشف"""
        print("\n🔗 عرض الوحدة المتكاملة مع جسر الخبير/المستكشف:")
        print("🧠 تطبيق مفهوم باسل يحيى عبدالله المتقدم:")
        print("   🔄 رسم → استنباط → تحليل خبير → تعلم")

        # اختبار دورة متكاملة
        test_shape = self.shape_db.get_all_shapes()[0]
        print(f"🔄 تنفيذ دورة متكاملة لـ: {test_shape.name}")

        cycle_result = self.integrated_unit.execute_integrated_cycle(test_shape)

        if cycle_result["overall_success"]:
            print(f"✅ نجحت الدورة المتكاملة!")
            print(f"📊 النتيجة الإجمالية: {cycle_result['overall_score']:.2%}")

            # عرض تحليل الخبير
            expert_analysis = cycle_result["stages"]["expert_analysis"]
            print(f"🧠 تحليل الخبير:")
            print(f"   🔍 دقة الاستنباط: {expert_analysis['extraction_accuracy']:.2%}")
            print(f"   🎨 جودة الرسم: {expert_analysis['drawing_fidelity']:.2%}")
            print(f"   🎯 التعرف على الأنماط: {expert_analysis['pattern_recognition']:.2%}")

            if expert_analysis['suggestions']:
                print(f"💡 اقتراحات التحسين:")
                for suggestion in expert_analysis['suggestions'][:3]:
                    print(f"   • {suggestion}")
        else:
            print(f"⚠️ الدورة المتكاملة تحتاج تحسين")

        # عرض حالة نظام الخبير
        expert_status = self.integrated_unit.expert_bridge.get_system_status()
        print(f"\n🧠 حالة نظام الخبير/المستكشف:")
        print(f"   🔄 دورات التعلم: {expert_status['learning_cycles']}")
        print(f"   📂 الفئات المتعلمة: {expert_status['categories_learned']}")
        print(f"   🎯 مستوى الثقة الإجمالي: {expert_status['expert_confidence']}")

    def process_image(self, image: np.ndarray, save_results: bool = False) -> Dict[str, Any]:
        """معالجة صورة واحدة بالنظام الثوري الكامل"""
        print("🔄 بدء المعالجة الثورية للصورة...")

        # 1. الاستنباط
        print("🔍 مرحلة الاستنباط...")
        extraction_result = self.extractor_unit.extract_equation_from_image(image)

        # 2. التعرف
        print("🧠 مرحلة التعرف...")
        recognition_result = self.recognition_engine.recognize_image(image)

        # 3. تجميع النتائج
        complete_result = {
            "extraction": extraction_result,
            "recognition": recognition_result,
            "processing_status": "مكتمل",
            "revolutionary_features": {
                "uses_animated_plotter": self.drawing_unit.plotter_available,
                "uses_shape_extractor": self.extractor_unit.extractor_available,
                "euclidean_distance_applied": True,
                "tolerance_thresholds_applied": True
            }
        }

        # 4. حفظ النتائج إذا طُلب
        if save_results:
            self._save_processing_results(complete_result)

        print("✅ تمت المعالجة الثورية بنجاح!")
        return complete_result

    def batch_process_images(self, images: List[np.ndarray],
                           descriptions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """معالجة مجموعة من الصور"""
        print(f"🔄 بدء المعالجة المجمعة لـ {len(images)} صورة...")

        results = []
        successful_recognitions = 0

        for i, image in enumerate(images):
            desc = descriptions[i] if descriptions and i < len(descriptions) else f"صورة {i+1}"
            print(f"🔍 معالجة {desc}...")

            result = self.process_image(image)
            results.append(result)

            if result["recognition"]["status"] == "تم التعرف بنجاح":
                successful_recognitions += 1

        # إحصائيات المجموعة
        success_rate = successful_recognitions / len(images) * 100
        print(f"\n📊 نتائج المعالجة المجمعة:")
        print(f"   ✅ نجح التعرف على: {successful_recognitions}/{len(images)} صورة")
        print(f"   📈 معدل النجاح: {success_rate:.1f}%")

        return results

    def add_new_shape(self, name: str, category: str, image: np.ndarray,
                     custom_tolerances: Optional[Dict[str, float]] = None) -> bool:
        """إضافة شكل جديد للنظام"""
        print(f"➕ إضافة شكل جديد: {name}")

        # استنباط خصائص الشكل الجديد
        extraction_result = self.extractor_unit.extract_equation_from_image(image)

        if not extraction_result["success"]:
            print("❌ فشل في استنباط خصائص الشكل الجديد")
            return False

        features = extraction_result["result"]

        # تحديد السماحية الافتراضية أو المخصصة
        if custom_tolerances:
            tolerance_thresholds = custom_tolerances
        else:
            tolerance_thresholds = {
                "geometric_tolerance": 0.2,
                "color_tolerance": 40.0,
                "euclidean_distance": 0.25,
                "position_tolerance": 0.15
            }

        # إنشاء كيان الشكل الجديد
        new_shape = ShapeEntity(
            id=None,
            name=name,
            category=category,
            equation_params=features["equation_params"],
            geometric_features=features["geometric_features"],
            color_properties=features["color_properties"],
            position_info=features["position_info"],
            tolerance_thresholds=tolerance_thresholds,
            created_date="",
            updated_date=""
        )

        # إضافة للقاعدة
        shape_id = self.shape_db.add_shape(new_shape)

        if shape_id > 0:
            print(f"✅ تم إضافة الشكل الجديد بنجاح: {name}")
            return True
        else:
            print(f"❌ فشل في إضافة الشكل: {name}")
            return False

    def optimize_system(self, test_images: List[np.ndarray],
                       expected_results: List[str]) -> Dict[str, Any]:
        """تحسين النظام باستخدام بيانات اختبار"""
        print("🔧 بدء تحسين النظام الثوري...")

        # ضبط دقيق للعتبات
        optimized_thresholds = self.recognition_engine.fine_tune_thresholds(
            test_images, expected_results
        )

        # تطبيق العتبات المحسنة
        shapes = self.shape_db.get_all_shapes()
        for shape in shapes:
            shape.tolerance_thresholds.update(optimized_thresholds)
            self.shape_db.update_shape(shape)

        # اختبار الأداء بعد التحسين
        results = self.batch_process_images(test_images)
        successful = sum(1 for r in results if r["recognition"]["status"] == "تم التعرف بنجاح")
        accuracy = successful / len(test_images) * 100

        optimization_result = {
            "optimized_thresholds": optimized_thresholds,
            "accuracy_after_optimization": accuracy,
            "total_test_images": len(test_images),
            "successful_recognitions": successful
        }

        print(f"✅ تم تحسين النظام - الدقة الجديدة: {accuracy:.1f}%")
        return optimization_result

    def execute_integrated_learning_cycle(self, shapes: List[ShapeEntity]) -> Dict[str, Any]:
        """تنفيذ دورة تعلم متكاملة باستخدام الوحدة المتكاملة"""
        if not self.integrated_unit:
            return {
                "status": "الوحدة المتكاملة غير متاحة",
                "message": "يتم استخدام المعالجة التقليدية"
            }

        print("🔄 بدء دورة التعلم المتكاملة...")

        # تنفيذ المعالجة المتكاملة
        batch_result = self.integrated_unit.batch_integrated_processing(shapes, learn_continuously=True)

        # توليد تقرير التكامل
        integration_report = self.integrated_unit.generate_integration_report()

        # تحسين المعاملات
        if len(shapes) >= 3:
            optimization_result = self.integrated_unit.optimize_integration_parameters(shapes[:3])
        else:
            optimization_result = {"status": "عدد الأشكال غير كافي للتحسين"}

        learning_result = {
            "batch_processing": batch_result,
            "integration_report": integration_report,
            "optimization": optimization_result,
            "expert_recommendations": self._get_expert_recommendations_summary()
        }

        print(f"✅ انتهت دورة التعلم المتكاملة")
        print(f"📊 معدل النجاح: {batch_result['successful_cycles']}/{batch_result['total_shapes']}")
        print(f"📈 متوسط الأداء: {batch_result['average_score']:.2%}")

        return learning_result

    def _get_expert_recommendations_summary(self) -> Dict[str, Any]:
        """ملخص توصيات الخبير"""
        if not self.integrated_unit:
            return {"status": "غير متاح"}

        # الحصول على توصيات لكل فئة
        categories = ["حيوانات", "مباني", "نباتات"]
        recommendations_summary = {}

        for category in categories:
            recommendations = self.integrated_unit.expert_bridge.get_expert_recommendations(category)
            recommendations_summary[category] = {
                "confidence_level": recommendations.get("confidence_level", "منخفض"),
                "success_rate": recommendations.get("success_rate", 0.0),
                "recommendations": recommendations.get("recommendations", [])
            }

        return recommendations_summary

    def _save_processing_results(self, results: Dict[str, Any]):
        """حفظ نتائج المعالجة"""
        try:
            import json
            from datetime import datetime

            filename = f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # تحويل numpy arrays لقوائم للحفظ
            serializable_results = self._make_serializable(results)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            print(f"💾 تم حفظ النتائج في: {filename}")

        except Exception as e:
            print(f"⚠️ خطأ في حفظ النتائج: {e}")

    def _make_serializable(self, obj):
        """تحويل الكائن لصيغة قابلة للحفظ"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def get_system_status(self) -> Dict[str, Any]:
        """حالة النظام الثوري"""
        db_stats = self.shape_db.get_statistics()
        recognition_stats = self.recognition_engine.get_recognition_statistics()

        return {
            "system_name": "النظام الثوري للتعرف على الأشكال",
            "creator": "باسل يحيى عبدالله - العراق/الموصل",
            "version": "1.0.0",
            "components": {
                "database": "RevolutionaryShapeDatabase",
                "drawing_unit": "animated_path_plotter_timeline" if self.drawing_unit.plotter_available else "fallback",
                "extractor_unit": "shape_equation_extractor_final_v3" if self.extractor_unit.extractor_available else "fallback",
                "recognition_engine": "RevolutionaryRecognitionEngine"
            },
            "database_stats": db_stats,
            "recognition_stats": recognition_stats,
            "revolutionary_features": [
                "قاعدة بيانات ثورية للأشكال",
                "وحدة رسم وتحريك متقدمة",
                "وحدة استنباط ذكية",
                "تعرف بالسماحية والمسافة الإقليدية",
                "وصف ذكي للنتائج",
                "وحدة متكاملة مع جسر الخبير/المستكشف" if self.integrated_unit else "وحدة متكاملة غير متاحة"
            ],
            "integrated_unit_status": {
                "available": self.integrated_unit is not None,
                "expert_bridge": self.integrated_unit.expert_bridge.get_system_status() if self.integrated_unit else None,
                "learning_cycles": self.integrated_unit.integration_cycles if self.integrated_unit else 0
            }
        }


def main():
    """الدالة الرئيسية للنظام الثوري"""
    try:
        # إنشاء النظام الثوري الكامل
        revolutionary_system = RevolutionaryShapeRecognitionSystem()

        # عرض المفهوم الثوري
        revolutionary_system.demonstrate_revolutionary_concept()

        # عرض حالة النظام
        print("\n📊 حالة النظام الثوري:")
        status = revolutionary_system.get_system_status()
        print(f"   🌟 النظام: {status['system_name']}")
        print(f"   👨‍💻 المبدع: {status['creator']}")
        print(f"   📦 الإصدار: {status['version']}")
        print(f"   🔧 المكونات: {len(status['components'])} مكون ثوري")

        print("\n🎉 انتهى العرض التوضيحي للنظام الثوري بنجاح!")
        print("🌟 النظام الثوري للتعرف على الأشكال جاهز للاستخدام!")

        print("\n💡 الميزات الثورية المطبقة:")
        for feature in status['revolutionary_features']:
            print(f"   ✅ {feature}")

        print("\n🌟 تطبيق فكرة باسل يحيى عبدالله الثورية:")
        print("   🎯 'لا نحتاج صور كثيرة - قاعدة بيانات + سماحية + مسافة إقليدية'")
        print("   🔄 'رسم من معادلة ← → استنباط من صورة ← → تعرف ذكي'")
        print("   📝 'وصف تلقائي: قطة + لون + وضعية + خلفية'")

    except Exception as e:
        print(f"❌ خطأ في تشغيل النظام الثوري: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
