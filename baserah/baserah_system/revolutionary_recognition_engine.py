#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Recognition Engine for Basira System
محرك التعرف الثوري - نظام بصيرة

Recognition engine implementing Basil Yahya Abdullah's revolutionary concept:
Smart recognition with tolerance thresholds and Euclidean distance.

محرك التعرف الذي يطبق مفهوم باسل يحيى عبدالله الثوري:
التعرف الذكي مع عتبات السماحية والمسافة الإقليدية.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import math
import sqlite3
from datetime import datetime
from typing import Dict, List, Any

from revolutionary_database import RevolutionaryShapeDatabase, ShapeEntity
from revolutionary_extractor_unit import RevolutionaryExtractorUnit


class RevolutionaryRecognitionEngine:
    """محرك التعرف الثوري مع السماحية والمسافة الإقليدية"""
    
    def __init__(self, shape_db: RevolutionaryShapeDatabase, 
                 extractor_unit: RevolutionaryExtractorUnit):
        """تهيئة محرك التعرف الثوري"""
        self.shape_db = shape_db
        self.extractor_unit = extractor_unit
        print("✅ تم تهيئة محرك التعرف الثوري")
    
    def recognize_image(self, image: np.ndarray) -> Dict[str, Any]:
        """التعرف الثوري على الصورة"""
        print("🔍 بدء التعرف الثوري على الصورة...")
        
        # 1. استنباط الخصائص من الصورة
        extraction_result = self.extractor_unit.extract_equation_from_image(image)
        
        if not extraction_result["success"]:
            return {
                "status": "فشل في الاستنباط",
                "confidence": 0.0,
                "message": "لم يتم استنباط الخصائص من الصورة"
            }
        
        extracted_features = extraction_result["result"]
        
        # 2. الحصول على جميع الأشكال المعروفة
        known_shapes = self.shape_db.get_all_shapes()
        
        # 3. حساب التشابه مع كل شكل
        recognition_candidates = []
        
        for shape in known_shapes:
            similarity_score = self._calculate_revolutionary_similarity(
                extracted_features, shape
            )
            
            # فحص السماحية
            within_tolerance = self._check_tolerance_thresholds(
                similarity_score, shape.tolerance_thresholds
            )
            
            recognition_candidates.append({
                "shape": shape,
                "similarity_score": similarity_score,
                "within_tolerance": within_tolerance,
                "euclidean_distance": similarity_score["euclidean_distance"],
                "geometric_match": similarity_score["geometric_similarity"],
                "color_match": similarity_score["color_similarity"]
            })
        
        # 4. ترتيب النتائج حسب أفضل تطابق
        recognition_candidates.sort(key=lambda x: x["euclidean_distance"])
        
        # 5. تحديد أفضل تطابق
        best_match = recognition_candidates[0] if recognition_candidates else None
        
        if best_match and best_match["within_tolerance"]:
            # تم التعرف بنجاح
            confidence = self._calculate_confidence(best_match["similarity_score"])
            
            # تسجيل في تاريخ التعرف
            self._log_recognition(image, best_match["shape"], confidence, 
                                best_match["euclidean_distance"])
            
            return {
                "status": "تم التعرف بنجاح",
                "recognized_shape": best_match["shape"].name,
                "category": best_match["shape"].category,
                "confidence": confidence,
                "euclidean_distance": best_match["euclidean_distance"],
                "geometric_similarity": best_match["geometric_match"],
                "color_similarity": best_match["color_match"],
                "extraction_method": extraction_result["method"],
                "description": self._generate_description(best_match["shape"], 
                                                        recognition_candidates),
                "all_candidates": recognition_candidates[:5]  # أفضل 5
            }
        else:
            # لم يتم التعرف
            return {
                "status": "لم يتم التعرف",
                "confidence": 0.0,
                "closest_match": best_match["shape"].name if best_match else "غير محدد",
                "euclidean_distance": best_match["euclidean_distance"] if best_match else float('inf'),
                "extraction_method": extraction_result["method"],
                "all_candidates": recognition_candidates[:5],
                "message": "الصورة خارج نطاق السماحية المقبولة"
            }
    
    def _calculate_revolutionary_similarity(self, extracted_features: Dict[str, Any], 
                                          known_shape: ShapeEntity) -> Dict[str, float]:
        """حساب التشابه الثوري"""
        
        # 1. التشابه الهندسي
        geometric_sim = self._calculate_geometric_similarity(
            extracted_features.get("geometric_features", {}),
            known_shape.geometric_features
        )
        
        # 2. التشابه اللوني
        color_sim = self._calculate_color_similarity(
            extracted_features.get("color_properties", {}),
            known_shape.color_properties
        )
        
        # 3. التشابه الموضعي
        position_sim = self._calculate_position_similarity(
            extracted_features.get("position_info", {}),
            known_shape.position_info
        )
        
        # 4. المسافة الإقليدية المجمعة (تطبيق فكرة باسل يحيى عبدالله)
        euclidean_distance = math.sqrt(
            geometric_sim**2 * 0.5 +
            color_sim**2 * 0.3 +
            position_sim**2 * 0.2
        )
        
        return {
            "geometric_similarity": geometric_sim,
            "color_similarity": color_sim,
            "position_similarity": position_sim,
            "euclidean_distance": euclidean_distance
        }
    
    def _calculate_geometric_similarity(self, extracted: Dict[str, float], 
                                      known: Dict[str, float]) -> float:
        """حساب التشابه الهندسي"""
        differences = []
        
        common_features = ["area", "perimeter", "aspect_ratio", "roundness", "compactness"]
        
        for feature in common_features:
            if feature in extracted and feature in known:
                if known[feature] != 0:
                    diff = abs(extracted[feature] - known[feature]) / abs(known[feature])
                else:
                    diff = abs(extracted[feature])
                differences.append(diff)
        
        return np.mean(differences) if differences else 1.0
    
    def _calculate_color_similarity(self, extracted: Dict[str, Any], 
                                   known: Dict[str, Any]) -> float:
        """حساب التشابه اللوني"""
        if "dominant_color" not in extracted or "dominant_color" not in known:
            return 1.0
        
        extracted_color = np.array(extracted["dominant_color"])
        known_color = np.array(known["dominant_color"])
        
        # المسافة الإقليدية في فضاء RGB
        color_distance = np.linalg.norm(extracted_color - known_color)
        
        # تطبيع (0-1)
        normalized_distance = color_distance / (255 * math.sqrt(3))
        
        return normalized_distance
    
    def _calculate_position_similarity(self, extracted: Dict[str, float], 
                                     known: Dict[str, float]) -> float:
        """حساب التشابه الموضعي"""
        if "center_x" not in extracted or "center_x" not in known:
            return 0.5
        
        pos_diff_x = abs(extracted["center_x"] - known["center_x"])
        pos_diff_y = abs(extracted["center_y"] - known["center_y"])
        
        return math.sqrt(pos_diff_x**2 + pos_diff_y**2)
    
    def _check_tolerance_thresholds(self, similarity_score: Dict[str, float], 
                                   thresholds: Dict[str, float]) -> bool:
        """فحص عتبات السماحية - تطبيق فكرة باسل يحيى عبدالله"""
        
        # فحص السماحية الهندسية
        geometric_ok = (similarity_score["geometric_similarity"] <= 
                       thresholds.get("geometric_tolerance", 0.2))
        
        # فحص السماحية اللونية
        color_ok = (similarity_score["color_similarity"] <= 
                   thresholds.get("color_tolerance", 50.0) / 255.0)
        
        # فحص المسافة الإقليدية (الفكرة الثورية)
        euclidean_ok = (similarity_score["euclidean_distance"] <= 
                       thresholds.get("euclidean_distance", 0.3))
        
        # يجب أن تكون جميع الشروط محققة
        return geometric_ok and color_ok and euclidean_ok
    
    def _calculate_confidence(self, similarity_score: Dict[str, float]) -> float:
        """حساب مستوى الثقة"""
        # كلما قلت المسافة الإقليدية، زادت الثقة
        euclidean_dist = similarity_score["euclidean_distance"]
        
        # تحويل المسافة إلى نسبة ثقة (0-1)
        confidence = max(0.0, 1.0 - euclidean_dist)
        
        return min(1.0, confidence)
    
    def _generate_description(self, recognized_shape: ShapeEntity, 
                            all_candidates: List[Dict]) -> str:
        """توليد وصف ذكي للنتيجة - تطبيق فكرة باسل يحيى عبدالله"""
        
        # تحليل السياق
        categories_found = set()
        colors_found = set()
        postures_found = set()
        
        for candidate in all_candidates[:3]:  # أفضل 3
            if candidate["within_tolerance"]:
                categories_found.add(candidate["shape"].category)
                
                # تحليل الألوان
                color = candidate["shape"].color_properties["dominant_color"]
                if color[0] > 200 and color[1] > 200 and color[2] > 200:
                    colors_found.add("أبيض")
                elif color[0] < 50 and color[1] < 50 and color[2] < 50:
                    colors_found.add("أسود")
                elif color[1] > color[0] and color[1] > color[2]:
                    colors_found.add("أخضر")
                elif color[0] > color[1] and color[0] > color[2]:
                    colors_found.add("أحمر")
                elif color[2] > color[0] and color[2] > color[1]:
                    colors_found.add("أزرق")
                
                # تحليل الوضعيات
                if "واقفة" in candidate["shape"].name:
                    postures_found.add("واقفة")
                elif "نائمة" in candidate["shape"].name:
                    postures_found.add("نائمة")
                elif "جالسة" in candidate["shape"].name:
                    postures_found.add("جالسة")
        
        # بناء الوصف الذكي (تطبيق فكرة باسل يحيى عبدالله)
        description = f"هذا {recognized_shape.name}"
        
        # إضافة معلومات الوضعية
        if postures_found:
            posture = list(postures_found)[0]
            if posture not in description:
                description += f" {posture}"
        
        # إضافة معلومات السياق
        if len(categories_found) > 1:
            other_categories = [cat for cat in categories_found if cat != recognized_shape.category]
            if other_categories:
                if "أشجار" in recognized_shape.name or any("شجرة" in cat for cat in other_categories):
                    description += " بخلفية أشجار"
                elif "بيوت" in recognized_shape.name or any("مباني" in cat for cat in other_categories):
                    description += " بخلفية بيوت"
                else:
                    description += f" في مشهد يحتوي على {', '.join(other_categories)}"
        
        # إضافة معلومات الألوان
        if len(colors_found) > 1:
            description += f" وألوان {', '.join(colors_found)}"
        
        return description
    
    def _log_recognition(self, image: np.ndarray, shape: ShapeEntity, 
                        confidence: float, similarity_score: float):
        """تسجيل عملية التعرف في قاعدة البيانات"""
        try:
            # حساب hash للصورة
            image_hash = str(hash(image.tobytes()))
            
            conn = sqlite3.connect(self.shape_db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO recognition_history (input_image_hash, recognized_shape_id,
                                           confidence_score, similarity_score, recognition_date)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                image_hash,
                shape.id,
                confidence,
                similarity_score,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ خطأ في تسجيل التعرف: {e}")
    
    def get_recognition_statistics(self) -> Dict[str, Any]:
        """إحصائيات التعرف"""
        try:
            conn = sqlite3.connect(self.shape_db.db_path)
            cursor = conn.cursor()
            
            # إجمالي عمليات التعرف
            cursor.execute('SELECT COUNT(*) FROM recognition_history')
            total_recognitions = cursor.fetchone()[0]
            
            # متوسط الثقة
            cursor.execute('SELECT AVG(confidence_score) FROM recognition_history')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # الأشكال الأكثر تعرفاً
            cursor.execute('''
            SELECT rs.name, COUNT(*) as count
            FROM recognition_history rh
            JOIN revolutionary_shapes rs ON rh.recognized_shape_id = rs.id
            GROUP BY rs.name
            ORDER BY count DESC
            LIMIT 5
            ''')
            top_shapes = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_recognitions": total_recognitions,
                "average_confidence": avg_confidence,
                "top_recognized_shapes": top_shapes
            }
            
        except Exception as e:
            print(f"⚠️ خطأ في جلب الإحصائيات: {e}")
            return {}
    
    def batch_recognize(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """التعرف على مجموعة من الصور"""
        results = []
        
        print(f"🔍 بدء التعرف على {len(images)} صورة...")
        
        for i, image in enumerate(images):
            print(f"🔍 معالجة الصورة {i+1}/{len(images)}...")
            result = self.recognize_image(image)
            results.append(result)
        
        # إحصائيات المجموعة
        successful_recognitions = sum(1 for r in results if r["status"] == "تم التعرف بنجاح")
        avg_confidence = np.mean([r["confidence"] for r in results if r["confidence"] > 0])
        
        print(f"✅ تم التعرف بنجاح على {successful_recognitions}/{len(images)} صورة")
        print(f"📈 متوسط الثقة: {avg_confidence:.2%}")
        
        return results
    
    def fine_tune_thresholds(self, test_images: List[np.ndarray], 
                           expected_results: List[str]) -> Dict[str, float]:
        """ضبط دقيق لعتبات السماحية"""
        print("🔧 بدء الضبط الدقيق لعتبات السماحية...")
        
        best_thresholds = {
            "geometric_tolerance": 0.2,
            "color_tolerance": 50.0,
            "euclidean_distance": 0.3
        }
        
        best_accuracy = 0.0
        
        # تجريب قيم مختلفة للعتبات
        for geo_tol in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for color_tol in [30.0, 40.0, 50.0, 60.0, 70.0]:
                for eucl_dist in [0.2, 0.25, 0.3, 0.35, 0.4]:
                    
                    # تطبيق العتبات الجديدة مؤقتاً
                    temp_accuracy = self._test_thresholds(
                        test_images, expected_results,
                        geo_tol, color_tol, eucl_dist
                    )
                    
                    if temp_accuracy > best_accuracy:
                        best_accuracy = temp_accuracy
                        best_thresholds = {
                            "geometric_tolerance": geo_tol,
                            "color_tolerance": color_tol,
                            "euclidean_distance": eucl_dist
                        }
        
        print(f"✅ أفضل عتبات: {best_thresholds}")
        print(f"📈 أفضل دقة: {best_accuracy:.2%}")
        
        return best_thresholds
    
    def _test_thresholds(self, test_images: List[np.ndarray], 
                        expected_results: List[str],
                        geo_tol: float, color_tol: float, eucl_dist: float) -> float:
        """اختبار عتبات معينة"""
        correct = 0
        
        for image, expected in zip(test_images, expected_results):
            # تعديل العتبات مؤقتاً
            shapes = self.shape_db.get_all_shapes()
            for shape in shapes:
                shape.tolerance_thresholds.update({
                    "geometric_tolerance": geo_tol,
                    "color_tolerance": color_tol,
                    "euclidean_distance": eucl_dist
                })
            
            result = self.recognize_image(image)
            if result["status"] == "تم التعرف بنجاح" and result["recognized_shape"] == expected:
                correct += 1
        
        return correct / len(test_images) if test_images else 0.0


def main():
    """اختبار محرك التعرف الثوري"""
    print("🧪 اختبار محرك التعرف الثوري...")
    
    # إنشاء المكونات
    from revolutionary_database import RevolutionaryShapeDatabase
    from revolutionary_extractor_unit import RevolutionaryExtractorUnit
    
    shape_db = RevolutionaryShapeDatabase()
    extractor_unit = RevolutionaryExtractorUnit()
    recognition_engine = RevolutionaryRecognitionEngine(shape_db, extractor_unit)
    
    # إنشاء صورة اختبار
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    # رسم شكل بسيط
    test_image[50:150, 50:150] = [255, 255, 255]  # مربع أبيض
    
    # اختبار التعرف
    print("🔍 اختبار التعرف...")
    result = recognition_engine.recognize_image(test_image)
    
    print(f"📊 النتيجة: {result['status']}")
    if result['status'] == "تم التعرف بنجاح":
        print(f"🎯 الشكل: {result['recognized_shape']}")
        print(f"📈 الثقة: {result['confidence']:.2%}")
        print(f"📏 المسافة الإقليدية: {result['euclidean_distance']:.4f}")
        print(f"📝 الوصف: {result['description']}")
    
    # عرض الإحصائيات
    stats = recognition_engine.get_recognition_statistics()
    print(f"\n📊 إحصائيات التعرف:")
    print(f"   إجمالي العمليات: {stats.get('total_recognitions', 0)}")
    print(f"   متوسط الثقة: {stats.get('average_confidence', 0):.2%}")


if __name__ == "__main__":
    main()
