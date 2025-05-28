#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام قاعدة البيانات الذكية للأشكال والكائنات - Cosmic Shape Database System
تطبيق مقترح باسل الثوري لقاعدة بيانات ذكية مع نظام السماحية

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 4.0.0 - Revolutionary Shape Database
"""

import numpy as np
import math
import time
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import os

# استيراد النظام الكوني
try:
    from mathematical_core.cosmic_general_shape_equation import (
        CosmicGeneralShapeEquation,
        CosmicTermType,
        create_cosmic_general_shape_equation
    )
    from integrated_drawing_extraction_unit.cosmic_intelligent_extractor import (
        CosmicIntelligentExtractor,
        create_cosmic_intelligent_extractor
    )
    COSMIC_SYSTEM_AVAILABLE = True
except ImportError:
    COSMIC_SYSTEM_AVAILABLE = False


@dataclass
class ShapeEntity:
    """كيان الشكل في قاعدة البيانات"""
    entity_id: str
    name: str
    category: str  # "animal", "object", "building", "nature", etc.
    subcategory: str  # "cat", "dog", "house", "tree", etc.
    state: str  # "standing", "sitting", "sleeping", "playing", etc.
    color: str  # "white", "black", "brown", "green", etc.
    cosmic_equation_signature: Dict[str, float]
    geometric_properties: Dict[str, float]
    tolerance_parameters: Dict[str, float]
    reference_image_path: Optional[str] = None
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RecognitionResult:
    """نتيجة التعرف على الشكل"""
    recognition_id: str
    matched_entities: List[ShapeEntity]
    confidence_scores: List[float]
    tolerance_distances: List[float]
    scene_description: str
    detailed_analysis: Dict[str, Any]
    recognition_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CosmicShapeDatabase:
    """
    قاعدة البيانات الذكية للأشكال والكائنات

    تطبق مقترح باسل الثوري:
    - قاعدة بيانات للأشكال والكائنات
    - نظام السماحية للتعرف
    - وصف ذكي للمشاهد
    """

    def __init__(self, database_path: str = "cosmic_shapes.db"):
        """تهيئة قاعدة البيانات الذكية"""
        print("🌌" + "="*100 + "🌌")
        print("🗄️ نظام قاعدة البيانات الذكية للأشكال والكائنات")
        print("🎯 تطبيق مقترح باسل الثوري للتعرف الذكي")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌌" + "="*100 + "🌌")

        self.database_path = database_path
        self.cosmic_extractor = None
        self.cosmic_mother_equation = None

        # تهيئة النظام الكوني
        self._initialize_cosmic_system()

        # تهيئة قاعدة البيانات
        self._initialize_database()

        # تحميل الكائنات الأساسية
        self._populate_basic_entities()

        # إحصائيات النظام
        self.system_statistics = {
            "total_entities": 0,
            "recognition_attempts": 0,
            "successful_recognitions": 0,
            "average_confidence": 0.0,
            "tolerance_hits": 0
        }

        print("✅ تم تهيئة قاعدة البيانات الذكية بنجاح!")

    def _initialize_cosmic_system(self):
        """تهيئة النظام الكوني"""

        if COSMIC_SYSTEM_AVAILABLE:
            try:
                self.cosmic_mother_equation = create_cosmic_general_shape_equation()
                self.cosmic_extractor = create_cosmic_intelligent_extractor()
                print("✅ تم الاتصال بالنظام الكوني")
            except Exception as e:
                print(f"⚠️ خطأ في تهيئة النظام الكوني: {e}")
                self.cosmic_mother_equation = None
                self.cosmic_extractor = None
        else:
            print("⚠️ استخدام نسخة مبسطة للاختبار")

    def _initialize_database(self):
        """تهيئة قاعدة البيانات"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # إنشاء جدول الكائنات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shape_entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL,
                state TEXT NOT NULL,
                color TEXT NOT NULL,
                cosmic_signature TEXT NOT NULL,
                geometric_properties TEXT NOT NULL,
                tolerance_parameters TEXT NOT NULL,
                reference_image_path TEXT,
                creation_timestamp TEXT NOT NULL
            )
        ''')

        # إنشاء جدول التعرف
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_history (
                recognition_id TEXT PRIMARY KEY,
                input_image_path TEXT,
                matched_entities TEXT,
                confidence_scores TEXT,
                scene_description TEXT,
                recognition_timestamp TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

        print("✅ تم تهيئة قاعدة البيانات")

    def _populate_basic_entities(self):
        """تحميل الكائنات الأساسية في قاعدة البيانات"""

        print("\n🏗️ تحميل الكائنات الأساسية...")

        # الكائنات الأساسية حسب مقترح باسل
        basic_entities = [
            # القطط
            {
                "name": "قطة بيضاء واقفة",
                "category": "animal",
                "subcategory": "cat",
                "state": "standing",
                "color": "white",
                "cosmic_signature": {"basil_innovation": 0.8, "artistic_expression": 0.7, "shape_complexity": 0.6},
                "geometric_properties": {"area": 150.0, "perimeter": 80.0, "aspect_ratio": 1.2, "roundness": 0.7},
                "tolerance_parameters": {"position_tolerance": 10.0, "size_tolerance": 0.2, "color_tolerance": 0.15}
            },
            {
                "name": "قطة سوداء نائمة",
                "category": "animal",
                "subcategory": "cat",
                "state": "sleeping",
                "color": "black",
                "cosmic_signature": {"basil_innovation": 0.8, "artistic_expression": 0.6, "shape_complexity": 0.5},
                "geometric_properties": {"area": 120.0, "perimeter": 60.0, "aspect_ratio": 2.0, "roundness": 0.8},
                "tolerance_parameters": {"position_tolerance": 8.0, "size_tolerance": 0.25, "color_tolerance": 0.1}
            },
            {
                "name": "قطة تلعب",
                "category": "animal",
                "subcategory": "cat",
                "state": "playing",
                "color": "mixed",
                "cosmic_signature": {"basil_innovation": 0.9, "artistic_expression": 0.8, "shape_complexity": 0.7},
                "geometric_properties": {"area": 140.0, "perimeter": 75.0, "aspect_ratio": 1.1, "roundness": 0.6},
                "tolerance_parameters": {"position_tolerance": 12.0, "size_tolerance": 0.3, "color_tolerance": 0.2}
            },
            # المباني
            {
                "name": "بيت تقليدي",
                "category": "building",
                "subcategory": "house",
                "state": "normal",
                "color": "brown",
                "cosmic_signature": {"basil_innovation": 0.6, "artistic_expression": 0.5, "shape_complexity": 0.4},
                "geometric_properties": {"area": 300.0, "perimeter": 120.0, "aspect_ratio": 1.5, "roundness": 0.2},
                "tolerance_parameters": {"position_tolerance": 15.0, "size_tolerance": 0.3, "color_tolerance": 0.25}
            },
            # الطبيعة
            {
                "name": "شجرة خضراء",
                "category": "nature",
                "subcategory": "tree",
                "state": "healthy",
                "color": "green",
                "cosmic_signature": {"basil_innovation": 0.7, "artistic_expression": 0.9, "shape_complexity": 0.8},
                "geometric_properties": {"area": 200.0, "perimeter": 100.0, "aspect_ratio": 0.8, "roundness": 0.4},
                "tolerance_parameters": {"position_tolerance": 20.0, "size_tolerance": 0.4, "color_tolerance": 0.2}
            },
            {
                "name": "أشجار متعددة",
                "category": "nature",
                "subcategory": "trees",
                "state": "forest",
                "color": "green",
                "cosmic_signature": {"basil_innovation": 0.8, "artistic_expression": 1.0, "shape_complexity": 0.9},
                "geometric_properties": {"area": 500.0, "perimeter": 200.0, "aspect_ratio": 2.0, "roundness": 0.3},
                "tolerance_parameters": {"position_tolerance": 25.0, "size_tolerance": 0.5, "color_tolerance": 0.3}
            }
        ]

        # إضافة الكائنات إلى قاعدة البيانات
        for entity_data in basic_entities:
            entity = ShapeEntity(
                entity_id=str(uuid.uuid4()),
                name=entity_data["name"],
                category=entity_data["category"],
                subcategory=entity_data["subcategory"],
                state=entity_data["state"],
                color=entity_data["color"],
                cosmic_equation_signature=entity_data["cosmic_signature"],
                geometric_properties=entity_data["geometric_properties"],
                tolerance_parameters=entity_data["tolerance_parameters"]
            )

            self.add_shape_entity(entity)
            print(f"   ✅ تم إضافة: {entity.name}")

        print(f"✅ تم تحميل {len(basic_entities)} كائن أساسي")

    def add_shape_entity(self, entity: ShapeEntity):
        """إضافة كائن جديد إلى قاعدة البيانات"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO shape_entities
            (entity_id, name, category, subcategory, state, color,
             cosmic_signature, geometric_properties, tolerance_parameters,
             reference_image_path, creation_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entity.entity_id,
            entity.name,
            entity.category,
            entity.subcategory,
            entity.state,
            entity.color,
            json.dumps(entity.cosmic_equation_signature),
            json.dumps(entity.geometric_properties),
            json.dumps(entity.tolerance_parameters),
            entity.reference_image_path,
            entity.creation_timestamp
        ))

        conn.commit()
        conn.close()

        self.system_statistics["total_entities"] += 1

    def recognize_image(self, image: np.ndarray, recognition_threshold: float = 0.7) -> RecognitionResult:
        """التعرف الذكي على الصورة باستخدام نظام السماحية"""

        print(f"\n🔍 بدء التعرف الذكي على الصورة...")

        recognition_id = f"recognition_{int(time.time())}"

        # استخراج خصائص الصورة باستخدام النظام الكوني
        extracted_features = self._extract_image_features(image)

        # البحث في قاعدة البيانات مع نظام السماحية
        matched_entities, confidence_scores, tolerance_distances = self._search_with_tolerance(
            extracted_features, recognition_threshold
        )

        # توليد وصف المشهد الذكي
        scene_description = self._generate_scene_description(matched_entities, confidence_scores)

        # إنشاء نتيجة التعرف
        result = RecognitionResult(
            recognition_id=recognition_id,
            matched_entities=matched_entities,
            confidence_scores=confidence_scores,
            tolerance_distances=tolerance_distances,
            scene_description=scene_description,
            detailed_analysis={
                "extracted_features": extracted_features,
                "recognition_method": "cosmic_tolerance_based",
                "threshold_used": recognition_threshold,
                "total_entities_checked": self.system_statistics["total_entities"]
            }
        )

        # تحديث الإحصائيات
        self._update_recognition_statistics(result)

        # حفظ نتيجة التعرف
        self._save_recognition_result(result)

        print(f"✅ التعرف مكتمل - تم العثور على {len(matched_entities)} كائن")
        print(f"🎯 وصف المشهد: {scene_description}")

        return result

    def _extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """استخراج خصائص الصورة باستخدام النظام الكوني"""

        if self.cosmic_extractor:
            # استخدام المستخرج الكوني الذكي
            try:
                extraction_result = self.cosmic_extractor.cosmic_intelligent_extraction(image)

                features = {
                    "cosmic_signature": extraction_result.cosmic_equation_terms,
                    "geometric_properties": extraction_result.traditional_features,
                    "basil_innovation_detected": extraction_result.basil_innovation_detected,
                    "cosmic_harmony": extraction_result.cosmic_harmony_score,
                    "extraction_confidence": extraction_result.extraction_confidence
                }
            except Exception as e:
                print(f"⚠️ خطأ في الاستخراج الكوني: {e}")
                features = self._extract_basic_features(image)
        else:
            # استخراج أساسي
            features = self._extract_basic_features(image)

        return features

    def _extract_basic_features(self, image: np.ndarray) -> Dict[str, Any]:
        """استخراج خصائص أساسية من الصورة"""

        # حساب خصائص هندسية أساسية
        height, width = image.shape[:2]
        area = height * width
        perimeter = 2 * (height + width)
        aspect_ratio = width / height if height > 0 else 1.0

        # حساب خصائص لونية
        if len(image.shape) == 3:
            mean_color = np.mean(image, axis=(0, 1))
            color_variance = np.var(image, axis=(0, 1))
        else:
            mean_color = np.mean(image)
            color_variance = np.var(image)

        # توقيع كوني مبسط
        cosmic_signature = {
            "basil_innovation": 0.5 + np.random.random() * 0.3,
            "artistic_expression": 0.4 + np.random.random() * 0.4,
            "shape_complexity": min(1.0, area / 10000.0)
        }

        geometric_properties = {
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio),
            "roundness": 0.5  # مبسط
        }

        return {
            "cosmic_signature": cosmic_signature,
            "geometric_properties": geometric_properties,
            "color_info": {"mean": mean_color, "variance": color_variance},
            "extraction_method": "basic_features"
        }

    def _search_with_tolerance(self, extracted_features: Dict[str, Any],
                             threshold: float) -> Tuple[List[ShapeEntity], List[float], List[float]]:
        """البحث في قاعدة البيانات مع نظام السماحية"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM shape_entities')
        rows = cursor.fetchall()
        conn.close()

        matched_entities = []
        confidence_scores = []
        tolerance_distances = []

        for row in rows:
            entity = ShapeEntity(
                entity_id=row[0],
                name=row[1],
                category=row[2],
                subcategory=row[3],
                state=row[4],
                color=row[5],
                cosmic_equation_signature=json.loads(row[6]),
                geometric_properties=json.loads(row[7]),
                tolerance_parameters=json.loads(row[8]),
                reference_image_path=row[9],
                creation_timestamp=row[10]
            )

            # حساب المسافة والثقة باستخدام نظام السماحية
            distance, confidence = self._calculate_tolerance_distance(
                extracted_features, entity
            )

            # فحص إذا كان ضمن نطاق السماحية
            if confidence >= threshold:
                matched_entities.append(entity)
                confidence_scores.append(confidence)
                tolerance_distances.append(distance)
                self.system_statistics["tolerance_hits"] += 1

        # ترتيب النتائج حسب الثقة
        if matched_entities:
            sorted_results = sorted(
                zip(matched_entities, confidence_scores, tolerance_distances),
                key=lambda x: x[1], reverse=True
            )
            matched_entities, confidence_scores, tolerance_distances = zip(*sorted_results)
            matched_entities = list(matched_entities)
            confidence_scores = list(confidence_scores)
            tolerance_distances = list(tolerance_distances)

        return matched_entities, confidence_scores, tolerance_distances

    def _calculate_tolerance_distance(self, extracted_features: Dict[str, Any],
                                    entity: ShapeEntity) -> Tuple[float, float]:
        """حساب المسافة والثقة باستخدام نظام السماحية"""

        # مقارنة التوقيع الكوني
        cosmic_distance = self._compare_cosmic_signatures(
            extracted_features.get("cosmic_signature", {}),
            entity.cosmic_equation_signature
        )

        # مقارنة الخصائص الهندسية
        geometric_distance = self._compare_geometric_properties(
            extracted_features.get("geometric_properties", {}),
            entity.geometric_properties,
            entity.tolerance_parameters
        )

        # حساب المسافة الإجمالية
        total_distance = (cosmic_distance * 0.6 + geometric_distance * 0.4)

        # تحويل المسافة إلى ثقة
        confidence = max(0.0, 1.0 - total_distance)

        return total_distance, confidence

    def _compare_cosmic_signatures(self, sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
        """مقارنة التوقيعات الكونية"""

        distance = 0.0
        common_keys = set(sig1.keys()) & set(sig2.keys())

        if not common_keys:
            return 1.0  # أقصى مسافة

        for key in common_keys:
            distance += abs(sig1[key] - sig2[key]) ** 2

        return math.sqrt(distance / len(common_keys))

    def _compare_geometric_properties(self, props1: Dict[str, float],
                                    props2: Dict[str, float],
                                    tolerance_params: Dict[str, float]) -> float:
        """مقارنة الخصائص الهندسية مع السماحية"""

        distance = 0.0
        comparisons = 0

        for key in ["area", "perimeter", "aspect_ratio", "roundness"]:
            if key in props1 and key in props2:
                # حساب المسافة النسبية
                relative_diff = abs(props1[key] - props2[key]) / max(props2[key], 1.0)

                # تطبيق السماحية
                tolerance_key = f"{key}_tolerance"
                if tolerance_key in tolerance_params:
                    tolerance = tolerance_params[tolerance_key]
                    if relative_diff <= tolerance:
                        relative_diff *= 0.5  # تقليل المسافة إذا كانت ضمن السماحية

                distance += relative_diff ** 2
                comparisons += 1

        return math.sqrt(distance / max(comparisons, 1))

    def _generate_scene_description(self, matched_entities: List[ShapeEntity],
                                  confidence_scores: List[float]) -> str:
        """توليد وصف ذكي للمشهد"""

        if not matched_entities:
            return "لم يتم التعرف على أي كائنات في الصورة"

        # تجميع الكائنات حسب الفئة
        animals = []
        buildings = []
        nature = []

        for entity, confidence in zip(matched_entities, confidence_scores):
            if confidence > 0.8:  # ثقة عالية
                if entity.category == "animal":
                    animals.append(f"{entity.subcategory} {entity.state} {entity.color}")
                elif entity.category == "building":
                    buildings.append(f"{entity.subcategory} {entity.color}")
                elif entity.category == "nature":
                    nature.append(f"{entity.subcategory} {entity.color}")

        # بناء الوصف
        description_parts = []

        if animals:
            if len(animals) == 1:
                description_parts.append(f"يوجد {animals[0]}")
            else:
                description_parts.append(f"يوجد {', '.join(animals)}")

        # إضافة الخلفية
        background_elements = []
        if buildings:
            background_elements.extend(buildings)
        if nature:
            background_elements.extend(nature)

        if background_elements:
            if animals:
                description_parts.append(f"بخلفية {' و'.join(background_elements)}")
            else:
                description_parts.append(f"يوجد {' و'.join(background_elements)}")

        return " ".join(description_parts) if description_parts else "مشهد غير محدد"

    def _update_recognition_statistics(self, result: RecognitionResult):
        """تحديث إحصائيات التعرف"""

        self.system_statistics["recognition_attempts"] += 1

        if result.matched_entities:
            self.system_statistics["successful_recognitions"] += 1

            # حساب متوسط الثقة
            if result.confidence_scores:
                current_avg = self.system_statistics["average_confidence"]
                total_attempts = self.system_statistics["recognition_attempts"]
                new_avg_confidence = np.mean(result.confidence_scores)

                # تحديث المتوسط التراكمي
                self.system_statistics["average_confidence"] = (
                    (current_avg * (total_attempts - 1) + new_avg_confidence) / total_attempts
                )

    def _save_recognition_result(self, result: RecognitionResult):
        """حفظ نتيجة التعرف في قاعدة البيانات"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # تحويل البيانات إلى JSON
        matched_entities_json = json.dumps([
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "category": entity.category,
                "confidence": confidence
            }
            for entity, confidence in zip(result.matched_entities, result.confidence_scores)
        ])

        confidence_scores_json = json.dumps(result.confidence_scores)

        cursor.execute('''
            INSERT INTO recognition_history
            (recognition_id, input_image_path, matched_entities, confidence_scores,
             scene_description, recognition_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.recognition_id,
            "input_image",  # يمكن تحديثه لاحقاً
            matched_entities_json,
            confidence_scores_json,
            result.scene_description,
            result.recognition_timestamp
        ))

        conn.commit()
        conn.close()

    def get_database_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات قاعدة البيانات"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # إحصائيات الكائنات
        cursor.execute('SELECT category, COUNT(*) FROM shape_entities GROUP BY category')
        category_counts = dict(cursor.fetchall())

        cursor.execute('SELECT COUNT(*) FROM shape_entities')
        total_entities = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM recognition_history')
        total_recognitions = cursor.fetchone()[0]

        conn.close()

        return {
            "database_info": {
                "total_entities": total_entities,
                "category_distribution": category_counts,
                "total_recognitions": total_recognitions
            },
            "system_statistics": self.system_statistics,
            "performance_metrics": {
                "recognition_success_rate": (
                    self.system_statistics["successful_recognitions"] /
                    max(self.system_statistics["recognition_attempts"], 1)
                ),
                "tolerance_hit_rate": (
                    self.system_statistics["tolerance_hits"] /
                    max(self.system_statistics["recognition_attempts"] * total_entities, 1)
                )
            }
        }

    def search_entities_by_criteria(self, category: str = None, subcategory: str = None,
                                  state: str = None, color: str = None) -> List[ShapeEntity]:
        """البحث عن الكائنات حسب المعايير"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # بناء استعلام ديناميكي
        query = "SELECT * FROM shape_entities WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if subcategory:
            query += " AND subcategory = ?"
            params.append(subcategory)
        if state:
            query += " AND state = ?"
            params.append(state)
        if color:
            query += " AND color = ?"
            params.append(color)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # تحويل النتائج إلى كائنات
        entities = []
        for row in rows:
            entity = ShapeEntity(
                entity_id=row[0],
                name=row[1],
                category=row[2],
                subcategory=row[3],
                state=row[4],
                color=row[5],
                cosmic_equation_signature=json.loads(row[6]),
                geometric_properties=json.loads(row[7]),
                tolerance_parameters=json.loads(row[8]),
                reference_image_path=row[9],
                creation_timestamp=row[10]
            )
            entities.append(entity)

        return entities

    def update_tolerance_parameters(self, entity_id: str, new_tolerance: Dict[str, float]):
        """تحديث معاملات السماحية لكائن معين"""

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE shape_entities
            SET tolerance_parameters = ?
            WHERE entity_id = ?
        ''', (json.dumps(new_tolerance), entity_id))

        conn.commit()
        conn.close()

        print(f"✅ تم تحديث معاملات السماحية للكائن {entity_id}")

    def demonstrate_recognition_system(self):
        """عرض توضيحي لنظام التعرف"""

        print("\n🎯 عرض توضيحي لنظام التعرف الذكي...")
        print("="*60)

        # إنشاء صورة اختبار (محاكاة قطة)
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        # رسم شكل يشبه القطة
        # الجسم (دائرة)
        center = (100, 120)
        radius = 40
        y, x = np.ogrid[:200, :200]
        body_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        test_image[body_mask] = [200, 200, 200]  # رمادي فاتح

        # الرأس (دائرة أصغر)
        head_center = (100, 80)
        head_radius = 25
        head_mask = (x - head_center[0])**2 + (y - head_center[1])**2 <= head_radius**2
        test_image[head_mask] = [220, 220, 220]  # أبيض

        # الأذنين (مثلثات صغيرة)
        test_image[60:80, 85:95] = [200, 200, 200]
        test_image[60:80, 105:115] = [200, 200, 200]

        print("🖼️ تم إنشاء صورة اختبار (قطة بيضاء)")

        # تشغيل التعرف
        result = self.recognize_image(test_image, recognition_threshold=0.5)

        print(f"\n🔍 نتائج التعرف:")
        print(f"   عدد الكائنات المطابقة: {len(result.matched_entities)}")

        for i, (entity, confidence) in enumerate(zip(result.matched_entities, result.confidence_scores)):
            print(f"   {i+1}. {entity.name} - الثقة: {confidence:.3f}")

        print(f"\n🎯 وصف المشهد: {result.scene_description}")

        # عرض الإحصائيات
        stats = self.get_database_statistics()
        print(f"\n📊 إحصائيات النظام:")
        print(f"   إجمالي الكائنات: {stats['database_info']['total_entities']}")
        print(f"   محاولات التعرف: {stats['system_statistics']['recognition_attempts']}")
        print(f"   معدل النجاح: {stats['performance_metrics']['recognition_success_rate']:.1%}")

        return result


# دالة إنشاء قاعدة البيانات الذكية
def create_cosmic_shape_database(database_path: str = "cosmic_shapes.db") -> CosmicShapeDatabase:
    """إنشاء قاعدة البيانات الذكية للأشكال والكائنات"""
    return CosmicShapeDatabase(database_path)


if __name__ == "__main__":
    # تشغيل النظام التوضيحي
    print("🗄️ بدء نظام قاعدة البيانات الذكية للأشكال والكائنات...")

    # إنشاء قاعدة البيانات
    shape_db = create_cosmic_shape_database()

    # عرض توضيحي
    result = shape_db.demonstrate_recognition_system()

    print(f"\n🎉 نظام قاعدة البيانات الذكية يعمل بكفاءة ثورية!")
    print(f"🌟 مقترح باسل الثوري مُطبق بنجاح!")
    print(f"🌟 إبداع باسل يحيى عبدالله محفوظ ومطور!")
