#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Drawing Unit for Basira System
وحدة الرسم الثورية - نظام بصيرة

Drawing component that uses animated_path_plotter_timeline for shape generation.
مكون الرسم الذي يستخدم animated_path_plotter_timeline لتوليد الأشكال.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, '.')

# محاولة استيراد الوحدة الأصلية
try:
    from animated_path_plotter_timeline import AnimatedPathPlotter
    DRAWING_UNIT_AVAILABLE = True
    print("✅ وحدة الرسم والتحريك متاحة: animated_path_plotter_timeline")
except ImportError as e:
    DRAWING_UNIT_AVAILABLE = False
    print(f"⚠️ وحدة الرسم والتحريك غير متاحة: {e}")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV غير متاح - سيتم استخدام رسم مبسط")

from revolutionary_database import ShapeEntity


class RevolutionaryDrawingUnit:
    """وحدة الرسم والتحريك الثورية"""

    def __init__(self):
        """تهيئة وحدة الرسم الثورية"""
        self.plotter_available = DRAWING_UNIT_AVAILABLE

        if self.plotter_available:
            try:
                self.animated_plotter = AnimatedPathPlotter()
                print("✅ تم تهيئة وحدة الرسم والتحريك الثورية")
            except Exception as e:
                print(f"⚠️ خطأ في تهيئة وحدة الرسم: {e}")
                self.plotter_available = False

        if not self.plotter_available:
            print("⚠️ سيتم استخدام وحدة رسم مبسطة")

    def draw_shape_from_equation(self, shape: ShapeEntity) -> Dict[str, Any]:
        """رسم الشكل من المعادلة باستخدام الوحدة الأصلية"""

        if self.plotter_available:
            try:
                # استخدام الوحدة الأصلية للرسم والتحريك
                result = self._use_original_plotter(shape)
                return {
                    "success": True,
                    "method": "animated_path_plotter_timeline",
                    "result": result,
                    "message": f"تم رسم {shape.name} باستخدام الوحدة الأصلية"
                }
            except Exception as e:
                print(f"⚠️ خطأ في الوحدة الأصلية: {e}")
                return self._use_fallback_drawing(shape)
        else:
            return self._use_fallback_drawing(shape)

    def _use_original_plotter(self, shape: ShapeEntity) -> Any:
        """استخدام وحدة الرسم الأصلية"""
        # تحويل معاملات الشكل لصيغة مناسبة للوحدة الأصلية
        equation_data = {
            "type": shape.category,
            "name": shape.name,
            "parameters": shape.equation_params,
            "colors": shape.color_properties,
            "position": shape.position_info
        }

        # استدعاء الوحدة الأصلية
        return self.animated_plotter.plot_from_equation(equation_data)

    def _use_fallback_drawing(self, shape: ShapeEntity) -> Dict[str, Any]:
        """استخدام رسم احتياطي مبسط"""
        canvas_size = (400, 400, 3)
        canvas = np.zeros(canvas_size, dtype=np.uint8)

        # رسم مبسط حسب الفئة
        if shape.category == "حيوانات":
            canvas = self._draw_simple_animal(canvas, shape)
        elif shape.category == "مباني":
            canvas = self._draw_simple_building(canvas, shape)
        elif shape.category == "نباتات":
            canvas = self._draw_simple_plant(canvas, shape)

        return {
            "success": True,
            "method": "fallback_simple_drawing",
            "result": canvas,
            "message": f"تم رسم {shape.name} باستخدام الرسم المبسط"
        }

    def _draw_simple_animal(self, canvas: np.ndarray, shape: ShapeEntity) -> np.ndarray:
        """رسم حيوان مبسط"""
        color = tuple(shape.color_properties["dominant_color"])

        # رسم جسم بسيط
        center_x = int(canvas.shape[1] * shape.position_info["center_x"])
        center_y = int(canvas.shape[0] * shape.position_info["center_y"])

        if CV2_AVAILABLE:
            # رسم متقدم مع OpenCV
            # جسم القطة (بيضاوي)
            cv2.ellipse(canvas, (center_x, center_y), (60, 40), 0, 0, 360, color, -1)
            # رأس القطة
            cv2.circle(canvas, (center_x, center_y - 50), 35, color, -1)
            # أذنان
            cv2.circle(canvas, (center_x - 20, center_y - 70), 15, color, -1)
            cv2.circle(canvas, (center_x + 20, center_y - 70), 15, color, -1)
            # ذيل
            cv2.ellipse(canvas, (center_x - 80, center_y + 20), (20, 50), 45, 0, 360, color, -1)

            # إضافة تفاصيل حسب الوضعية
            if "نائمة" in shape.name:
                # عيون مغلقة
                cv2.line(canvas, (center_x - 15, center_y - 50), (center_x - 5, center_y - 50), (0, 0, 0), 2)
                cv2.line(canvas, (center_x + 5, center_y - 50), (center_x + 15, center_y - 50), (0, 0, 0), 2)
            else:
                # عيون مفتوحة
                cv2.circle(canvas, (center_x - 10, center_y - 50), 5, (0, 0, 0), -1)
                cv2.circle(canvas, (center_x + 10, center_y - 50), 5, (0, 0, 0), -1)
        else:
            # رسم مبسط بدون OpenCV
            canvas[center_y-30:center_y+30, center_x-40:center_x+40] = color
            # رأس
            canvas[center_y-80:center_y-20, center_x-20:center_x+20] = color

        return canvas

    def _draw_simple_building(self, canvas: np.ndarray, shape: ShapeEntity) -> np.ndarray:
        """رسم مبنى مبسط"""
        color = tuple(shape.color_properties["dominant_color"])

        center_x = int(canvas.shape[1] * shape.position_info["center_x"])
        center_y = int(canvas.shape[0] * shape.position_info["center_y"])

        if CV2_AVAILABLE:
            # قاعدة البيت
            cv2.rectangle(canvas, (center_x - 50, center_y), (center_x + 50, center_y + 80), color, -1)
            # سقف البيت
            roof_color = (200, 100, 50)
            roof_points = np.array([[center_x - 60, center_y], [center_x, center_y - 40], [center_x + 60, center_y]], np.int32)
            cv2.fillPoly(canvas, [roof_points], roof_color)
            # باب
            cv2.rectangle(canvas, (center_x - 20, center_y + 30), (center_x + 20, center_y + 80), (100, 50, 25), -1)
            # نوافذ
            cv2.rectangle(canvas, (center_x - 40, center_y + 10), (center_x - 25, center_y + 25), (135, 206, 235), -1)
            cv2.rectangle(canvas, (center_x + 25, center_y + 10), (center_x + 40, center_y + 25), (135, 206, 235), -1)

            # إضافة أشجار في الخلفية إذا كان البيت "بخلفية أشجار"
            if "أشجار" in shape.name:
                tree_color = (34, 139, 34)
                # شجرة يسار
                cv2.rectangle(canvas, (center_x - 120, center_y + 40), (center_x - 110, center_y + 80), (101, 67, 33), -1)
                cv2.circle(canvas, (center_x - 115, center_y + 20), 25, tree_color, -1)
                # شجرة يمين
                cv2.rectangle(canvas, (center_x + 110, center_y + 40), (center_x + 120, center_y + 80), (101, 67, 33), -1)
                cv2.circle(canvas, (center_x + 115, center_y + 20), 25, tree_color, -1)
        else:
            # رسم مبسط
            canvas[center_y:center_y+80, center_x-50:center_x+50] = color

        return canvas

    def _draw_simple_plant(self, canvas: np.ndarray, shape: ShapeEntity) -> np.ndarray:
        """رسم نبات مبسط"""
        trunk_color = (101, 67, 33)
        leaves_color = tuple(shape.color_properties["dominant_color"])

        center_x = int(canvas.shape[1] * shape.position_info["center_x"])
        center_y = int(canvas.shape[0] * shape.position_info["center_y"])

        if CV2_AVAILABLE:
            # جذع الشجرة
            trunk_width = int(shape.equation_params.get("trunk_width", 0.15) * 100)
            trunk_height = int(shape.equation_params.get("trunk_height", 0.5) * 200)
            cv2.rectangle(canvas, (center_x - trunk_width//2, center_y),
                         (center_x + trunk_width//2, center_y + trunk_height), trunk_color, -1)

            # أوراق الشجرة
            crown_radius = int(shape.equation_params.get("crown_radius", 0.4) * 100)
            cv2.circle(canvas, (center_x, center_y - crown_radius//2), crown_radius, leaves_color, -1)

            # إضافة فروع إذا كانت شجرة كبيرة
            if "كبيرة" in shape.name:
                # فروع إضافية
                cv2.circle(canvas, (center_x - 30, center_y - 20), crown_radius//2, leaves_color, -1)
                cv2.circle(canvas, (center_x + 30, center_y - 20), crown_radius//2, leaves_color, -1)
        else:
            # رسم مبسط
            canvas[center_y:center_y+100, center_x-10:center_x+10] = trunk_color
            canvas[center_y-60:center_y+20, center_x-50:center_x+50] = leaves_color

        return canvas

    def create_animated_sequence(self, shape: ShapeEntity, frames: int = 30) -> List[np.ndarray]:
        """إنشاء تسلسل متحرك للشكل"""
        if self.plotter_available:
            try:
                # استخدام الوحدة الأصلية للتحريك
                return self.animated_plotter.create_animation_sequence(shape, frames)
            except Exception as e:
                print(f"⚠️ خطأ في التحريك الأصلي: {e}")

        # تحريك مبسط
        sequence = []
        for frame in range(frames):
            # تغيير بسيط في الموقع أو الحجم
            modified_shape = shape
            # يمكن إضافة تعديلات للتحريك هنا

            result = self._use_fallback_drawing(modified_shape)
            if result["success"]:
                sequence.append(result["result"])

        return sequence

    def save_shape_image(self, shape: ShapeEntity, filename: str) -> bool:
        """حفظ صورة الشكل"""
        try:
            result = self.draw_shape_from_equation(shape)
            if result["success"] and isinstance(result["result"], np.ndarray):
                if CV2_AVAILABLE:
                    cv2.imwrite(filename, result["result"])
                    print(f"✅ تم حفظ صورة {shape.name} في {filename}")
                    return True
                else:
                    print("⚠️ OpenCV غير متاح لحفظ الصورة")
                    return False
            return False
        except Exception as e:
            print(f"❌ خطأ في حفظ الصورة: {e}")
            return False


def main():
    """اختبار وحدة الرسم الثورية"""
    print("🧪 اختبار وحدة الرسم الثورية...")

    # إنشاء وحدة الرسم
    drawing_unit = RevolutionaryDrawingUnit()

    # إنشاء شكل اختبار
    from revolutionary_database import ShapeEntity

    test_shape = ShapeEntity(
        id=1,
        name="قطة اختبار",
        category="حيوانات",
        equation_params={
            "body_curve_a": 0.8,
            "head_radius": 0.3,
            "ear_angle": 45.0
        },
        geometric_features={
            "area": 150.0,
            "perimeter": 45.0
        },
        color_properties={
            "dominant_color": [255, 200, 100]
        },
        position_info={
            "center_x": 0.5,
            "center_y": 0.5
        },
        tolerance_thresholds={},
        created_date="",
        updated_date=""
    )

    # اختبار الرسم
    print("🖌️ اختبار رسم الشكل...")
    result = drawing_unit.draw_shape_from_equation(test_shape)

    print(f"📊 النتيجة: {result['success']}")
    print(f"🔧 الطريقة: {result['method']}")
    print(f"📝 الرسالة: {result['message']}")

    # اختبار حفظ الصورة
    if result["success"]:
        print("💾 اختبار حفظ الصورة...")
        drawing_unit.save_shape_image(test_shape, "test_shape.png")


if __name__ == "__main__":
    main()
