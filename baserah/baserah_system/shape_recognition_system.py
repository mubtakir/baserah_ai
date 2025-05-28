#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Shape Recognition System for Basira
نظام التعرف الثوري على الأشكال لنظام بصيرة

This system implements Basil Yahya Abdullah's revolutionary concept:
1. Database of shapes with names, equations, and properties
2. Drawing unit: equation → shape
3. Reverse engineering unit: image/data → equation  
4. Smart recognition: compare with tolerance thresholds

هذا النظام يطبق مفهوم باسل يحيى عبدالله الثوري:
1. قاعدة بيانات الأشكال مع الأسماء والمعادلات والخصائص
2. وحدة الرسم: معادلة ← شكل
3. وحدة الهندسة العكسية: صورة/بيانات ← معادلة
4. التعرف الذكي: مقارنة مع عتبات السماحية

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import json
import sqlite3
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV غير متاح - سيتم استخدام معالجة مبسطة")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib غير متاح - سيتم استخدام رسم مبسط")


@dataclass
class ShapeProperties:
    """خصائص الشكل"""
    name: str
    category: str
    equation_params: Dict[str, float]
    geometric_features: Dict[str, float]
    color_properties: Dict[str, Any]
    position_info: Dict[str, float]
    tolerance_thresholds: Dict[str, float]


class ShapeDatabase:
    """قاعدة بيانات الأشكال الثورية"""
    
    def __init__(self, db_path: str = "shapes_database.db"):
        """تهيئة قاعدة البيانات"""
        self.db_path = db_path
        self.init_database()
        self.load_default_shapes()
    
    def init_database(self):
        """إنشاء قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shapes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            equation_params TEXT,
            geometric_features TEXT,
            color_properties TEXT,
            position_info TEXT,
            tolerance_thresholds TEXT,
            created_date TEXT,
            updated_date TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ تم إنشاء قاعدة بيانات الأشكال")
    
    def add_shape(self, shape_props: ShapeProperties):
        """إضافة شكل جديد لقاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO shapes (name, category, equation_params, geometric_features,
                          color_properties, position_info, tolerance_thresholds,
                          created_date, updated_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            shape_props.name,
            shape_props.category,
            json.dumps(shape_props.equation_params),
            json.dumps(shape_props.geometric_features),
            json.dumps(shape_props.color_properties),
            json.dumps(shape_props.position_info),
            json.dumps(shape_props.tolerance_thresholds),
            current_time,
            current_time
        ))
        
        conn.commit()
        conn.close()
        print(f"✅ تم إضافة الشكل: {shape_props.name}")
    
    def get_all_shapes(self) -> List[ShapeProperties]:
        """الحصول على جميع الأشكال"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM shapes')
        rows = cursor.fetchall()
        
        shapes = []
        for row in rows:
            shape = ShapeProperties(
                name=row[1],
                category=row[2],
                equation_params=json.loads(row[3]),
                geometric_features=json.loads(row[4]),
                color_properties=json.loads(row[5]),
                position_info=json.loads(row[6]),
                tolerance_thresholds=json.loads(row[7])
            )
            shapes.append(shape)
        
        conn.close()
        return shapes
    
    def load_default_shapes(self):
        """تحميل الأشكال الافتراضية"""
        # فحص إذا كانت قاعدة البيانات فارغة
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM shapes')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print("📦 تحميل الأشكال الافتراضية...")
            self._create_default_shapes()
    
    def _create_default_shapes(self):
        """إنشاء الأشكال الافتراضية"""
        
        # قطة بيضاء
        white_cat = ShapeProperties(
            name="قطة بيضاء",
            category="حيوانات",
            equation_params={
                "body_curve": 0.8,
                "head_radius": 0.3,
                "ear_angle": 45,
                "tail_curve": 1.2
            },
            geometric_features={
                "area": 150.0,
                "perimeter": 45.0,
                "aspect_ratio": 1.4,
                "roundness": 0.7
            },
            color_properties={
                "dominant_color": [255, 255, 255],
                "secondary_colors": [[200, 200, 200], [180, 180, 180]],
                "brightness": 0.9
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.5,
                "orientation": 0.0
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.15,
                "color_tolerance": 30.0,
                "euclidean_distance": 0.2
            }
        )
        
        # قطة سوداء
        black_cat = ShapeProperties(
            name="قطة سوداء",
            category="حيوانات",
            equation_params={
                "body_curve": 0.8,
                "head_radius": 0.3,
                "ear_angle": 45,
                "tail_curve": 1.2
            },
            geometric_features={
                "area": 150.0,
                "perimeter": 45.0,
                "aspect_ratio": 1.4,
                "roundness": 0.7
            },
            color_properties={
                "dominant_color": [30, 30, 30],
                "secondary_colors": [[50, 50, 50], [70, 70, 70]],
                "brightness": 0.2
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.5,
                "orientation": 0.0
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.15,
                "color_tolerance": 30.0,
                "euclidean_distance": 0.2
            }
        )
        
        # بيت
        house = ShapeProperties(
            name="بيت",
            category="مباني",
            equation_params={
                "base_width": 1.0,
                "base_height": 0.8,
                "roof_angle": 30,
                "door_ratio": 0.3
            },
            geometric_features={
                "area": 200.0,
                "perimeter": 60.0,
                "aspect_ratio": 1.25,
                "roundness": 0.3
            },
            color_properties={
                "dominant_color": [150, 100, 80],
                "secondary_colors": [[200, 150, 100], [100, 80, 60]],
                "brightness": 0.6
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.4,
                "orientation": 0.0
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.2,
                "color_tolerance": 40.0,
                "euclidean_distance": 0.25
            }
        )
        
        # شجرة
        tree = ShapeProperties(
            name="شجرة",
            category="نباتات",
            equation_params={
                "trunk_height": 0.4,
                "trunk_width": 0.1,
                "crown_radius": 0.3,
                "branch_density": 0.8
            },
            geometric_features={
                "area": 120.0,
                "perimeter": 40.0,
                "aspect_ratio": 2.0,
                "roundness": 0.6
            },
            color_properties={
                "dominant_color": [34, 139, 34],
                "secondary_colors": [[101, 67, 33], [50, 205, 50]],
                "brightness": 0.5
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.3,
                "orientation": 0.0
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.25,
                "color_tolerance": 35.0,
                "euclidean_distance": 0.3
            }
        )
        
        # إضافة الأشكال لقاعدة البيانات
        for shape in [white_cat, black_cat, house, tree]:
            self.add_shape(shape)


class DrawingUnit:
    """وحدة الرسم - من المعادلة إلى الشكل"""
    
    def __init__(self):
        """تهيئة وحدة الرسم"""
        self.canvas_size = (400, 400)
        print("✅ تم تهيئة وحدة الرسم")
    
    def draw_from_equation(self, shape_props: ShapeProperties) -> np.ndarray:
        """رسم الشكل من المعادلة"""
        canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
        
        if shape_props.category == "حيوانات":
            canvas = self._draw_animal(canvas, shape_props)
        elif shape_props.category == "مباني":
            canvas = self._draw_building(canvas, shape_props)
        elif shape_props.category == "نباتات":
            canvas = self._draw_plant(canvas, shape_props)
        
        return canvas
    
    def _draw_animal(self, canvas: np.ndarray, shape_props: ShapeProperties) -> np.ndarray:
        """رسم حيوان"""
        color = tuple(shape_props.color_properties["dominant_color"])
        
        # رسم جسم القطة (دائرة بيضاوية)
        center = (200, 250)
        axes = (60, 40)
        
        if CV2_AVAILABLE:
            cv2.ellipse(canvas, center, axes, 0, 0, 360, color, -1)
            # رأس القطة
            cv2.circle(canvas, (200, 180), 35, color, -1)
            # أذنان
            cv2.circle(canvas, (180, 160), 15, color, -1)
            cv2.circle(canvas, (220, 160), 15, color, -1)
            # ذيل
            cv2.ellipse(canvas, (140, 260), (20, 50), 45, 0, 360, color, -1)
        else:
            # رسم مبسط بدون OpenCV
            canvas[160:220, 165:235] = color
        
        return canvas
    
    def _draw_building(self, canvas: np.ndarray, shape_props: ShapeProperties) -> np.ndarray:
        """رسم مبنى"""
        color = tuple(shape_props.color_properties["dominant_color"])
        
        if CV2_AVAILABLE:
            # قاعدة البيت
            cv2.rectangle(canvas, (150, 200), (250, 300), color, -1)
            # سقف البيت
            roof_points = np.array([[150, 200], [200, 150], [250, 200]], np.int32)
            cv2.fillPoly(canvas, [roof_points], (200, 100, 50))
            # باب
            cv2.rectangle(canvas, (180, 250), (220, 300), (100, 50, 25), -1)
        else:
            # رسم مبسط
            canvas[200:300, 150:250] = color
        
        return canvas
    
    def _draw_plant(self, canvas: np.ndarray, shape_props: ShapeProperties) -> np.ndarray:
        """رسم نبات"""
        trunk_color = (101, 67, 33)
        leaves_color = tuple(shape_props.color_properties["dominant_color"])
        
        if CV2_AVAILABLE:
            # جذع الشجرة
            cv2.rectangle(canvas, (190, 250), (210, 350), trunk_color, -1)
            # أوراق الشجرة
            cv2.circle(canvas, (200, 200), 50, leaves_color, -1)
        else:
            # رسم مبسط
            canvas[250:350, 190:210] = trunk_color
            canvas[150:250, 150:250] = leaves_color
        
        return canvas


class ReverseEngineeringUnit:
    """وحدة الهندسة العكسية - من الصورة إلى المعادلة"""
    
    def __init__(self):
        """تهيئة وحدة الهندسة العكسية"""
        print("✅ تم تهيئة وحدة الهندسة العكسية")
    
    def extract_features_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """استخراج الخصائص من الصورة"""
        features = {
            "geometric_features": self._extract_geometric_features(image),
            "color_properties": self._extract_color_properties(image),
            "position_info": self._extract_position_info(image)
        }
        
        return features
    
    def _extract_geometric_features(self, image: np.ndarray) -> Dict[str, float]:
        """استخراج الخصائص الهندسية"""
        # تحويل لرمادي
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # حساب المساحة (البكسلات غير السوداء)
        area = np.sum(gray > 50)
        
        # حساب المحيط (تقدير مبسط)
        edges = self._simple_edge_detection(gray)
        perimeter = np.sum(edges > 0)
        
        # نسبة العرض للارتفاع
        height, width = gray.shape
        aspect_ratio = width / height if height > 0 else 1.0
        
        # الاستدارة
        roundness = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": aspect_ratio,
            "roundness": roundness
        }
    
    def _extract_color_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """استخراج خصائص الألوان"""
        if len(image.shape) == 3:
            # اللون المهيمن
            pixels = image.reshape(-1, 3)
            # إزالة البكسلات السوداء
            non_black = pixels[np.sum(pixels, axis=1) > 30]
            
            if len(non_black) > 0:
                dominant_color = np.mean(non_black, axis=0).astype(int).tolist()
                brightness = np.mean(non_black) / 255.0
            else:
                dominant_color = [0, 0, 0]
                brightness = 0.0
        else:
            dominant_color = [128, 128, 128]
            brightness = np.mean(image) / 255.0
        
        return {
            "dominant_color": dominant_color,
            "secondary_colors": [],
            "brightness": brightness
        }
    
    def _extract_position_info(self, image: np.ndarray) -> Dict[str, float]:
        """استخراج معلومات الموقع"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # العثور على مركز الكتلة
        y_indices, x_indices = np.where(gray > 50)
        
        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = np.mean(x_indices) / gray.shape[1]
            center_y = np.mean(y_indices) / gray.shape[0]
        else:
            center_x = 0.5
            center_y = 0.5
        
        return {
            "center_x": center_x,
            "center_y": center_y,
            "orientation": 0.0
        }
    
    def _simple_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """كشف الحواف المبسط"""
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


class SmartRecognitionEngine:
    """محرك التعرف الذكي"""
    
    def __init__(self, shape_db: ShapeDatabase):
        """تهيئة محرك التعرف"""
        self.shape_db = shape_db
        self.reverse_unit = ReverseEngineeringUnit()
        print("✅ تم تهيئة محرك التعرف الذكي")
    
    def recognize_shape(self, image: np.ndarray) -> Dict[str, Any]:
        """التعرف على الشكل في الصورة"""
        # استخراج الخصائص من الصورة
        extracted_features = self.reverse_unit.extract_features_from_image(image)
        
        # الحصول على جميع الأشكال من قاعدة البيانات
        known_shapes = self.shape_db.get_all_shapes()
        
        # مقارنة مع كل شكل معروف
        best_match = None
        best_score = float('inf')
        recognition_results = []
        
        for shape in known_shapes:
            score = self._calculate_similarity_score(extracted_features, shape)
            
            recognition_results.append({
                "shape_name": shape.name,
                "category": shape.category,
                "similarity_score": score,
                "within_tolerance": score <= self._get_combined_tolerance(shape)
            })
            
            if score < best_score:
                best_score = score
                best_match = shape
        
        # تحديد النتيجة النهائية
        if best_match and best_score <= self._get_combined_tolerance(best_match):
            recognition_status = "تم التعرف بنجاح"
            confidence = max(0, 1 - (best_score / self._get_combined_tolerance(best_match)))
        else:
            recognition_status = "لم يتم التعرف"
            confidence = 0.0
        
        return {
            "status": recognition_status,
            "best_match": best_match.name if best_match else "غير معروف",
            "confidence": confidence,
            "similarity_score": best_score,
            "extracted_features": extracted_features,
            "all_results": recognition_results
        }
    
    def _calculate_similarity_score(self, extracted_features: Dict[str, Any], 
                                   known_shape: ShapeProperties) -> float:
        """حساب درجة التشابه"""
        geometric_score = self._geometric_similarity(
            extracted_features["geometric_features"],
            known_shape.geometric_features,
            known_shape.tolerance_thresholds["geometric_tolerance"]
        )
        
        color_score = self._color_similarity(
            extracted_features["color_properties"],
            known_shape.color_properties,
            known_shape.tolerance_thresholds["color_tolerance"]
        )
        
        position_score = self._position_similarity(
            extracted_features["position_info"],
            known_shape.position_info
        )
        
        # المتوسط المرجح
        total_score = (geometric_score * 0.5 + color_score * 0.3 + position_score * 0.2)
        
        return total_score
    
    def _geometric_similarity(self, extracted: Dict[str, float], 
                            known: Dict[str, float], tolerance: float) -> float:
        """حساب التشابه الهندسي"""
        differences = []
        
        for key in known.keys():
            if key in extracted:
                if known[key] != 0:
                    diff = abs(extracted[key] - known[key]) / known[key]
                else:
                    diff = abs(extracted[key])
                differences.append(diff)
        
        return np.mean(differences) if differences else 1.0
    
    def _color_similarity(self, extracted: Dict[str, Any], 
                         known: Dict[str, Any], tolerance: float) -> float:
        """حساب التشابه اللوني"""
        extracted_color = np.array(extracted["dominant_color"])
        known_color = np.array(known["dominant_color"])
        
        # المسافة الإقليدية بين الألوان
        color_distance = np.linalg.norm(extracted_color - known_color)
        
        # تطبيع النتيجة
        normalized_distance = color_distance / (255 * math.sqrt(3))
        
        return normalized_distance
    
    def _position_similarity(self, extracted: Dict[str, float], 
                           known: Dict[str, float]) -> float:
        """حساب التشابه الموضعي"""
        pos_diff_x = abs(extracted["center_x"] - known["center_x"])
        pos_diff_y = abs(extracted["center_y"] - known["center_y"])
        
        return math.sqrt(pos_diff_x**2 + pos_diff_y**2)
    
    def _get_combined_tolerance(self, shape: ShapeProperties) -> float:
        """حساب السماحية المجمعة"""
        return (shape.tolerance_thresholds["geometric_tolerance"] * 0.5 +
                shape.tolerance_thresholds["color_tolerance"] / 255.0 * 0.3 +
                shape.tolerance_thresholds["euclidean_distance"] * 0.2)


class RevolutionaryShapeRecognitionSystem:
    """النظام الثوري للتعرف على الأشكال"""
    
    def __init__(self):
        """تهيئة النظام الثوري"""
        print("🌟" + "="*80 + "🌟")
        print("🚀 النظام الثوري للتعرف على الأشكال - نظام بصيرة 🚀")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*80 + "🌟")
        
        # تهيئة المكونات
        self.shape_db = ShapeDatabase()
        self.drawing_unit = DrawingUnit()
        self.recognition_engine = SmartRecognitionEngine(self.shape_db)
        
        print("✅ تم تهيئة النظام الثوري بنجاح!")
    
    def demonstrate_system(self):
        """عرض توضيحي للنظام"""
        print("\n🎯 بدء العرض التوضيحي للنظام الثوري...")
        
        # 1. عرض قاعدة البيانات
        self._demonstrate_database()
        
        # 2. عرض وحدة الرسم
        self._demonstrate_drawing()
        
        # 3. عرض التعرف الذكي
        self._demonstrate_recognition()
    
    def _demonstrate_database(self):
        """عرض قاعدة البيانات"""
        print("\n📊 قاعدة بيانات الأشكال:")
        shapes = self.shape_db.get_all_shapes()
        
        for i, shape in enumerate(shapes, 1):
            print(f"{i}. {shape.name} ({shape.category})")
            print(f"   🎨 اللون المهيمن: {shape.color_properties['dominant_color']}")
            print(f"   📐 المساحة: {shape.geometric_features['area']}")
            print(f"   🎯 السماحية: {shape.tolerance_thresholds['euclidean_distance']}")
    
    def _demonstrate_drawing(self):
        """عرض وحدة الرسم"""
        print("\n🎨 عرض وحدة الرسم:")
        shapes = self.shape_db.get_all_shapes()
        
        for shape in shapes[:2]:  # رسم أول شكلين
            print(f"🖌️ رسم {shape.name}...")
            canvas = self.drawing_unit.draw_from_equation(shape)
            print(f"   ✅ تم رسم {shape.name} بنجاح!")
    
    def _demonstrate_recognition(self):
        """عرض التعرف الذكي"""
        print("\n🧠 عرض التعرف الذكي:")
        
        # إنشاء صورة اختبار
        test_shape = self.shape_db.get_all_shapes()[0]  # أول شكل
        test_image = self.drawing_unit.draw_from_equation(test_shape)
        
        print(f"🔍 اختبار التعرف على: {test_shape.name}")
        
        # تطبيق التعرف
        result = self.recognition_engine.recognize_shape(test_image)
        
        print(f"📊 النتيجة: {result['status']}")
        print(f"🎯 أفضل تطابق: {result['best_match']}")
        print(f"📈 الثقة: {result['confidence']:.2%}")
        print(f"📏 درجة التشابه: {result['similarity_score']:.4f}")


def main():
    """الدالة الرئيسية"""
    try:
        # إنشاء النظام الثوري
        system = RevolutionaryShapeRecognitionSystem()
        
        # تشغيل العرض التوضيحي
        system.demonstrate_system()
        
        print("\n🎉 انتهى العرض التوضيحي بنجاح!")
        print("🌟 النظام الثوري للتعرف على الأشكال جاهز للاستخدام!")
        
    except Exception as e:
        print(f"❌ خطأ في تشغيل النظام: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
