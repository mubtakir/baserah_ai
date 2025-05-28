#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Shape Database for Basira System
قاعدة البيانات الثورية للأشكال - نظام بصيرة

Database component of Basil Yahya Abdullah's revolutionary shape recognition concept.
مكون قاعدة البيانات من مفهوم باسل يحيى عبدالله الثوري للتعرف على الأشكال.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ShapeEntity:
    """كيان الشكل في قاعدة البيانات الثورية"""
    id: Optional[int]
    name: str
    category: str
    equation_params: Dict[str, float]
    geometric_features: Dict[str, float]
    color_properties: Dict[str, Any]
    position_info: Dict[str, float]
    tolerance_thresholds: Dict[str, float]
    created_date: str
    updated_date: str


class RevolutionaryShapeDatabase:
    """قاعدة البيانات الثورية للأشكال"""
    
    def __init__(self, db_path: str = "revolutionary_shapes.db"):
        """تهيئة قاعدة البيانات الثورية"""
        self.db_path = db_path
        self.init_database()
        self.load_revolutionary_shapes()
        print("✅ تم تهيئة قاعدة البيانات الثورية للأشكال")
    
    def init_database(self):
        """إنشاء قاعدة البيانات الثورية"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS revolutionary_shapes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
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
        
        # جدول لتاريخ التعرف
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_image_hash TEXT,
            recognized_shape_id INTEGER,
            confidence_score REAL,
            similarity_score REAL,
            recognition_date TEXT,
            FOREIGN KEY (recognized_shape_id) REFERENCES revolutionary_shapes (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ تم إنشاء قاعدة البيانات الثورية")
    
    def add_shape(self, shape: ShapeEntity) -> int:
        """إضافة شكل جديد لقاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        try:
            cursor.execute('''
            INSERT INTO revolutionary_shapes (name, category, equation_params, geometric_features,
                              color_properties, position_info, tolerance_thresholds,
                              created_date, updated_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                shape.name,
                shape.category,
                json.dumps(shape.equation_params),
                json.dumps(shape.geometric_features),
                json.dumps(shape.color_properties),
                json.dumps(shape.position_info),
                json.dumps(shape.tolerance_thresholds),
                current_time,
                current_time
            ))
            
            shape_id = cursor.lastrowid
            conn.commit()
            print(f"✅ تم إضافة الشكل الثوري: {shape.name}")
            return shape_id
            
        except sqlite3.IntegrityError:
            print(f"⚠️ الشكل {shape.name} موجود مسبقاً")
            return -1
        finally:
            conn.close()
    
    def get_all_shapes(self) -> List[ShapeEntity]:
        """الحصول على جميع الأشكال الثورية"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM revolutionary_shapes')
        rows = cursor.fetchall()
        
        shapes = []
        for row in rows:
            shape = ShapeEntity(
                id=row[0],
                name=row[1],
                category=row[2],
                equation_params=json.loads(row[3]),
                geometric_features=json.loads(row[4]),
                color_properties=json.loads(row[5]),
                position_info=json.loads(row[6]),
                tolerance_thresholds=json.loads(row[7]),
                created_date=row[8],
                updated_date=row[9]
            )
            shapes.append(shape)
        
        conn.close()
        return shapes
    
    def get_shape_by_name(self, name: str) -> Optional[ShapeEntity]:
        """الحصول على شكل بالاسم"""
        shapes = self.get_all_shapes()
        for shape in shapes:
            if shape.name == name:
                return shape
        return None
    
    def update_shape(self, shape: ShapeEntity) -> bool:
        """تحديث شكل موجود"""
        if not shape.id:
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            UPDATE revolutionary_shapes 
            SET name=?, category=?, equation_params=?, geometric_features=?,
                color_properties=?, position_info=?, tolerance_thresholds=?, updated_date=?
            WHERE id=?
            ''', (
                shape.name,
                shape.category,
                json.dumps(shape.equation_params),
                json.dumps(shape.geometric_features),
                json.dumps(shape.color_properties),
                json.dumps(shape.position_info),
                json.dumps(shape.tolerance_thresholds),
                datetime.now().isoformat(),
                shape.id
            ))
            
            conn.commit()
            print(f"✅ تم تحديث الشكل: {shape.name}")
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تحديث الشكل: {e}")
            return False
        finally:
            conn.close()
    
    def delete_shape(self, shape_id: int) -> bool:
        """حذف شكل"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM revolutionary_shapes WHERE id=?', (shape_id,))
            conn.commit()
            print(f"✅ تم حذف الشكل رقم: {shape_id}")
            return True
        except Exception as e:
            print(f"❌ خطأ في حذف الشكل: {e}")
            return False
        finally:
            conn.close()
    
    def load_revolutionary_shapes(self):
        """تحميل الأشكال الثورية الافتراضية"""
        # فحص إذا كانت قاعدة البيانات فارغة
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM revolutionary_shapes')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print("📦 تحميل الأشكال الثورية الافتراضية...")
            self._create_revolutionary_shapes()
    
    def _create_revolutionary_shapes(self):
        """إنشاء الأشكال الثورية الافتراضية"""
        
        # قطة بيضاء واقفة
        white_standing_cat = ShapeEntity(
            id=None,
            name="قطة بيضاء واقفة",
            category="حيوانات",
            equation_params={
                "body_curve_a": 0.8,
                "body_curve_b": 0.6,
                "head_radius": 0.3,
                "ear_angle": 45.0,
                "tail_curve": 1.2,
                "leg_length": 0.4,
                "posture": "standing"
            },
            geometric_features={
                "area": 150.0,
                "perimeter": 45.0,
                "aspect_ratio": 1.4,
                "roundness": 0.7,
                "compactness": 0.85,
                "elongation": 1.2
            },
            color_properties={
                "dominant_color": [255, 255, 255],
                "secondary_colors": [[200, 200, 200], [180, 180, 180]],
                "brightness": 0.9,
                "saturation": 0.1,
                "hue_range": [0, 360]
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.6,
                "orientation": 0.0,
                "bounding_box": [0.2, 0.3, 0.8, 0.9]
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.15,
                "color_tolerance": 30.0,
                "euclidean_distance": 0.2,
                "position_tolerance": 0.1
            },
            created_date="",
            updated_date=""
        )
        
        # قطة سوداء نائمة
        black_sleeping_cat = ShapeEntity(
            id=None,
            name="قطة سوداء نائمة",
            category="حيوانات",
            equation_params={
                "body_curve_a": 1.2,
                "body_curve_b": 0.4,
                "head_radius": 0.25,
                "ear_angle": 30.0,
                "tail_curve": 0.8,
                "leg_length": 0.1,
                "posture": "sleeping"
            },
            geometric_features={
                "area": 120.0,
                "perimeter": 38.0,
                "aspect_ratio": 2.1,
                "roundness": 0.9,
                "compactness": 0.95,
                "elongation": 0.8
            },
            color_properties={
                "dominant_color": [30, 30, 30],
                "secondary_colors": [[50, 50, 50], [70, 70, 70]],
                "brightness": 0.2,
                "saturation": 0.05,
                "hue_range": [0, 60]
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.7,
                "orientation": 90.0,
                "bounding_box": [0.1, 0.6, 0.9, 0.8]
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.18,
                "color_tolerance": 25.0,
                "euclidean_distance": 0.22,
                "position_tolerance": 0.15
            },
            created_date="",
            updated_date=""
        )
        
        # بيت بخلفية أشجار
        house_with_trees = ShapeEntity(
            id=None,
            name="بيت بخلفية أشجار",
            category="مباني",
            equation_params={
                "base_width": 1.0,
                "base_height": 0.8,
                "roof_angle": 30.0,
                "door_ratio": 0.3,
                "window_count": 2,
                "chimney_height": 0.2,
                "background_trees": 3
            },
            geometric_features={
                "area": 250.0,
                "perimeter": 65.0,
                "aspect_ratio": 1.25,
                "roundness": 0.3,
                "compactness": 0.7,
                "elongation": 1.0
            },
            color_properties={
                "dominant_color": [150, 100, 80],
                "secondary_colors": [[34, 139, 34], [200, 150, 100]],
                "brightness": 0.6,
                "saturation": 0.4,
                "hue_range": [20, 120]
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.4,
                "orientation": 0.0,
                "bounding_box": [0.2, 0.2, 0.8, 0.7]
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.2,
                "color_tolerance": 40.0,
                "euclidean_distance": 0.25,
                "position_tolerance": 0.12
            },
            created_date="",
            updated_date=""
        )
        
        # شجرة كبيرة
        large_tree = ShapeEntity(
            id=None,
            name="شجرة كبيرة",
            category="نباتات",
            equation_params={
                "trunk_height": 0.5,
                "trunk_width": 0.15,
                "crown_radius": 0.4,
                "branch_density": 0.9,
                "leaf_density": 0.8,
                "tree_age": "mature"
            },
            geometric_features={
                "area": 180.0,
                "perimeter": 50.0,
                "aspect_ratio": 1.8,
                "roundness": 0.6,
                "compactness": 0.8,
                "elongation": 1.5
            },
            color_properties={
                "dominant_color": [34, 139, 34],
                "secondary_colors": [[101, 67, 33], [50, 205, 50]],
                "brightness": 0.5,
                "saturation": 0.7,
                "hue_range": [90, 150]
            },
            position_info={
                "center_x": 0.5,
                "center_y": 0.3,
                "orientation": 0.0,
                "bounding_box": [0.3, 0.1, 0.7, 0.6]
            },
            tolerance_thresholds={
                "geometric_tolerance": 0.25,
                "color_tolerance": 35.0,
                "euclidean_distance": 0.3,
                "position_tolerance": 0.2
            },
            created_date="",
            updated_date=""
        )
        
        # إضافة الأشكال الثورية لقاعدة البيانات
        revolutionary_shapes = [
            white_standing_cat,
            black_sleeping_cat, 
            house_with_trees,
            large_tree
        ]
        
        for shape in revolutionary_shapes:
            self.add_shape(shape)
        
        print(f"✅ تم تحميل {len(revolutionary_shapes)} شكل ثوري افتراضي")
    
    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات قاعدة البيانات"""
        shapes = self.get_all_shapes()
        
        categories = {}
        for shape in shapes:
            if shape.category not in categories:
                categories[shape.category] = 0
            categories[shape.category] += 1
        
        return {
            "total_shapes": len(shapes),
            "categories": categories,
            "average_tolerance": sum(s.tolerance_thresholds["euclidean_distance"] for s in shapes) / len(shapes) if shapes else 0
        }


def main():
    """اختبار قاعدة البيانات الثورية"""
    print("🧪 اختبار قاعدة البيانات الثورية...")
    
    db = RevolutionaryShapeDatabase()
    
    # عرض الإحصائيات
    stats = db.get_statistics()
    print(f"\n📊 إحصائيات قاعدة البيانات:")
    print(f"   إجمالي الأشكال: {stats['total_shapes']}")
    print(f"   الفئات: {stats['categories']}")
    print(f"   متوسط السماحية: {stats['average_tolerance']:.3f}")
    
    # عرض جميع الأشكال
    shapes = db.get_all_shapes()
    print(f"\n📋 الأشكال الثورية:")
    for i, shape in enumerate(shapes, 1):
        print(f"{i}. {shape.name} ({shape.category})")
        print(f"   🎯 السماحية الإقليدية: {shape.tolerance_thresholds['euclidean_distance']}")


if __name__ == "__main__":
    main()
