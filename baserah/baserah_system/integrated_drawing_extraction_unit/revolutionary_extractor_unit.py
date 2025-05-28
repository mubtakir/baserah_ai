#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Extractor Unit for Basira System
وحدة الاستنباط الثورية - نظام بصيرة

Extraction component that uses shape_equation_extractor_final_v3 for reverse engineering.
مكون الاستنباط الذي يستخدم shape_equation_extractor_final_v3 للهندسة العكسية.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import math
import sys
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, '.')

# محاولة استيراد الوحدة الأصلية
try:
    from shape_equation_extractor_final_v3 import ShapeEquationExtractor
    EXTRACTOR_UNIT_AVAILABLE = True
    print("✅ وحدة المستنبط متاحة: shape_equation_extractor_final_v3")
except ImportError as e:
    EXTRACTOR_UNIT_AVAILABLE = False
    print(f"⚠️ وحدة المستنبط غير متاحة: {e}")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV غير متاح - سيتم استخدام معالجة مبسطة")


class RevolutionaryExtractorUnit:
    """وحدة الاستنباط الثورية"""

    def __init__(self):
        """تهيئة وحدة الاستنباط الثورية"""
        self.extractor_available = EXTRACTOR_UNIT_AVAILABLE

        if self.extractor_available:
            try:
                self.shape_extractor = ShapeEquationExtractor()
                print("✅ تم تهيئة وحدة الاستنباط الثورية")
            except Exception as e:
                print(f"⚠️ خطأ في تهيئة وحدة الاستنباط: {e}")
                self.extractor_available = False

        if not self.extractor_available:
            print("⚠️ سيتم استخدام وحدة استنباط مبسطة")

    def extract_equation_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """استنباط المعادلة من الصورة باستخدام الوحدة الأصلية"""

        if self.extractor_available:
            try:
                # استخدام الوحدة الأصلية للاستنباط
                result = self._use_original_extractor(image)
                return {
                    "success": True,
                    "method": "shape_equation_extractor_final_v3",
                    "result": result,
                    "message": "تم الاستنباط باستخدام الوحدة الأصلية"
                }
            except Exception as e:
                print(f"⚠️ خطأ في الوحدة الأصلية: {e}")
                return self._use_fallback_extraction(image)
        else:
            return self._use_fallback_extraction(image)

    def _use_original_extractor(self, image: np.ndarray) -> Dict[str, Any]:
        """استخدام وحدة الاستنباط الأصلية"""
        # استدعاء الوحدة الأصلية
        return self.shape_extractor.extract_shape_equation(image)

    def _use_fallback_extraction(self, image: np.ndarray) -> Dict[str, Any]:
        """استخدام استنباط احتياطي مبسط"""
        # استخراج خصائص مبسطة
        features = self._extract_simple_features(image)

        return {
            "success": True,
            "method": "fallback_simple_extraction",
            "message": "تم الاستنباط باستخدام الطريقة المبسطة",
            "result": {
                "equation_params": features["equation_params"],
                "geometric_features": features["geometric_features"],
                "color_properties": features["color_properties"],
                "position_info": features["position_info"],
                "confidence": 0.7
            }
        }

    def _extract_simple_features(self, image: np.ndarray) -> Dict[str, Any]:
        """استخراج خصائص مبسطة"""
        # تحويل لرمادي
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image

        # حساب الخصائص الأساسية
        area = np.sum(gray > 50)
        height, width = gray.shape

        # استخراج اللون المهيمن
        dominant_color = self._extract_dominant_color(image)

        # العثور على مركز الكتلة
        center_x, center_y = self._find_center_of_mass(gray)

        # حساب الخصائص الهندسية المتقدمة
        geometric_features = self._calculate_advanced_geometry(gray)

        # تقدير معاملات المعادلة
        equation_params = self._estimate_equation_parameters(gray, geometric_features)

        return {
            "equation_params": equation_params,
            "geometric_features": geometric_features,
            "color_properties": {
                "dominant_color": dominant_color,
                "secondary_colors": self._extract_secondary_colors(image),
                "brightness": np.mean(image) / 255.0 if len(image.shape) == 3 else np.mean(gray) / 255.0,
                "saturation": self._calculate_saturation(image),
                "hue_range": self._calculate_hue_range(image)
            },
            "position_info": {
                "center_x": center_x,
                "center_y": center_y,
                "orientation": self._calculate_orientation(gray),
                "bounding_box": self._find_bounding_box(gray)
            }
        }

    def _extract_dominant_color(self, image: np.ndarray) -> list:
        """استخراج اللون المهيمن"""
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
            # إزالة البكسلات السوداء
            non_black = pixels[np.sum(pixels, axis=1) > 30]

            if len(non_black) > 0:
                return np.mean(non_black, axis=0).astype(int).tolist()
            else:
                return [128, 128, 128]
        else:
            return [128, 128, 128]

    def _extract_secondary_colors(self, image: np.ndarray) -> list:
        """استخراج الألوان الثانوية"""
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
            non_black = pixels[np.sum(pixels, axis=1) > 30]

            if len(non_black) > 100:
                # تجميع الألوان باستخدام k-means مبسط
                unique_colors = []
                for i in range(0, len(non_black), len(non_black)//5):
                    color = non_black[i].tolist()
                    if not any(np.linalg.norm(np.array(color) - np.array(uc)) < 50 for uc in unique_colors):
                        unique_colors.append(color)
                        if len(unique_colors) >= 3:
                            break

                return unique_colors[:3]

        return []

    def _find_center_of_mass(self, gray: np.ndarray) -> tuple:
        """العثور على مركز الكتلة"""
        y_indices, x_indices = np.where(gray > 50)

        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = np.mean(x_indices) / gray.shape[1]
            center_y = np.mean(y_indices) / gray.shape[0]
        else:
            center_x = 0.5
            center_y = 0.5

        return center_x, center_y

    def _calculate_advanced_geometry(self, gray: np.ndarray) -> Dict[str, float]:
        """حساب الخصائص الهندسية المتقدمة"""
        # حساب المساحة
        area = float(np.sum(gray > 50))

        # حساب المحيط (تقدير مبسط)
        edges = self._simple_edge_detection(gray)
        perimeter = float(np.sum(edges > 0))

        # نسبة العرض للارتفاع
        height, width = gray.shape
        aspect_ratio = width / height if height > 0 else 1.0

        # الاستدارة
        roundness = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        # الكثافة
        compactness = area / (width * height) if (width * height) > 0 else 0

        # الاستطالة
        elongation = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0

        return {
            "area": area,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "roundness": max(0, min(1, roundness)),
            "compactness": max(0, min(1, compactness)),
            "elongation": elongation
        }

    def _estimate_equation_parameters(self, gray: np.ndarray, geometric_features: Dict[str, float]) -> Dict[str, float]:
        """تقدير معاملات المعادلة"""
        # تقدير المعاملات بناءً على الخصائص الهندسية
        roundness = geometric_features["roundness"]
        aspect_ratio = geometric_features["aspect_ratio"]
        elongation = geometric_features["elongation"]

        # تقدير منحنى الجسم
        body_curve_a = 0.5 + (roundness * 0.5)
        body_curve_b = 0.3 + (aspect_ratio * 0.3)

        # تقدير نصف القطر
        head_radius = 0.2 + (roundness * 0.2)

        # تقدير الزاوية
        ear_angle = 30.0 + (elongation * 20.0)

        # تقدير منحنى الذيل
        tail_curve = 0.8 + (aspect_ratio * 0.4)

        return {
            "body_curve_a": body_curve_a,
            "body_curve_b": body_curve_b,
            "head_radius": head_radius,
            "ear_angle": ear_angle,
            "tail_curve": tail_curve,
            "estimated_curve": roundness,
            "estimated_radius": head_radius,
            "estimated_angle": ear_angle
        }

    def _calculate_saturation(self, image: np.ndarray) -> float:
        """حساب التشبع اللوني"""
        if len(image.shape) == 3:
            # تحويل لـ HSV مبسط
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)

            saturation = np.where(max_val > 0, (max_val - min_val) / max_val, 0)
            return float(np.mean(saturation))

        return 0.0

    def _calculate_hue_range(self, image: np.ndarray) -> list:
        """حساب نطاق الصبغة"""
        if len(image.shape) == 3:
            # تقدير مبسط لنطاق الصبغة
            dominant_color = self._extract_dominant_color(image)
            r, g, b = dominant_color

            if g > r and g > b:
                return [60, 180]  # أخضر
            elif r > g and r > b:
                return [0, 60]    # أحمر
            elif b > r and b > g:
                return [180, 300] # أزرق
            else:
                return [0, 360]   # متنوع

        return [0, 360]

    def _calculate_orientation(self, gray: np.ndarray) -> float:
        """حساب الاتجاه"""
        # حساب الاتجاه الرئيسي للشكل
        y_indices, x_indices = np.where(gray > 50)

        if len(x_indices) > 1 and len(y_indices) > 1:
            # حساب الاتجاه باستخدام PCA مبسط
            center_x = np.mean(x_indices)
            center_y = np.mean(y_indices)

            # حساب التباين
            var_x = np.var(x_indices - center_x)
            var_y = np.var(y_indices - center_y)

            if var_x > var_y:
                return 0.0  # أفقي
            else:
                return 90.0  # عمودي

        return 0.0

    def _find_bounding_box(self, gray: np.ndarray) -> list:
        """العثور على المربع المحيط"""
        y_indices, x_indices = np.where(gray > 50)

        if len(x_indices) > 0 and len(y_indices) > 0:
            min_x = np.min(x_indices) / gray.shape[1]
            max_x = np.max(x_indices) / gray.shape[1]
            min_y = np.min(y_indices) / gray.shape[0]
            max_y = np.max(y_indices) / gray.shape[0]

            return [min_x, min_y, max_x, max_y]

        return [0.1, 0.1, 0.9, 0.9]

    def _simple_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """كشف الحواف المبسط"""
        if CV2_AVAILABLE:
            return cv2.Canny(gray, 50, 150)
        else:
            # مرشح Sobel مبسط
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # تطبيق المرشحات
            edges_x = self._convolve2d(gray, kernel_x)
            edges_y = self._convolve2d(gray, kernel_y)

            # حساب قوة الحافة
            edges = np.sqrt(edges_x**2 + edges_y**2)

            return edges

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """تطبيق مرشح ثنائي الأبعاد"""
        result = np.zeros_like(image, dtype=np.float32)

        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)

        return result

    def analyze_image_complexity(self, image: np.ndarray) -> Dict[str, float]:
        """تحليل تعقيد الصورة"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image

        # حساب التعقيد
        edges = self._simple_edge_detection(gray)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # حساب التنوع اللوني
        if len(image.shape) == 3:
            unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
            color_diversity = unique_colors / (image.shape[0] * image.shape[1])
        else:
            color_diversity = 0.0

        # حساب التماثل
        symmetry_score = self._calculate_symmetry(gray)

        return {
            "edge_density": edge_density,
            "color_diversity": color_diversity,
            "symmetry_score": symmetry_score,
            "complexity_score": (edge_density + color_diversity) / 2
        }

    def _calculate_symmetry(self, gray: np.ndarray) -> float:
        """حساب التماثل"""
        # تماثل أفقي
        left_half = gray[:, :gray.shape[1]//2]
        right_half = np.fliplr(gray[:, gray.shape[1]//2:])

        if left_half.shape == right_half.shape:
            horizontal_symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
        else:
            horizontal_symmetry = 0.0

        return max(0, horizontal_symmetry)


def main():
    """اختبار وحدة الاستنباط الثورية"""
    print("🧪 اختبار وحدة الاستنباط الثورية...")

    # إنشاء وحدة الاستنباط
    extractor_unit = RevolutionaryExtractorUnit()

    # إنشاء صورة اختبار
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    # رسم دائرة بسيطة
    center = (100, 100)
    radius = 50
    color = (255, 200, 100)

    y, x = np.ogrid[:200, :200]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    test_image[mask] = color

    # اختبار الاستنباط
    print("🔍 اختبار استنباط الخصائص...")
    result = extractor_unit.extract_equation_from_image(test_image)

    print(f"📊 النتيجة: {result['success']}")
    print(f"🔧 الطريقة: {result['method']}")
    print(f"📝 الرسالة: {result['message']}")

    if result["success"]:
        features = result["result"]
        print(f"🎨 اللون المهيمن: {features['color_properties']['dominant_color']}")
        print(f"📐 المساحة: {features['geometric_features']['area']:.1f}")
        print(f"🔄 الاستدارة: {features['geometric_features']['roundness']:.3f}")

    # اختبار تحليل التعقيد
    print("\n🔬 اختبار تحليل التعقيد...")
    complexity = extractor_unit.analyze_image_complexity(test_image)
    print(f"📊 كثافة الحواف: {complexity['edge_density']:.3f}")
    print(f"🌈 التنوع اللوني: {complexity['color_diversity']:.3f}")
    print(f"🔄 التماثل: {complexity['symmetry_score']:.3f}")


if __name__ == "__main__":
    main()
