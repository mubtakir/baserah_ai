#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Interactive Interface for Basira System
الواجهة التفاعلية الثورية - نظام بصيرة

Interactive command-line interface for Basil Yahya Abdullah's revolutionary system.
واجهة سطر الأوامر التفاعلية لنظام باسل يحيى عبدالله الثوري.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import numpy as np
import sys
import os
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, '.')

from revolutionary_system_unified import RevolutionaryShapeRecognitionSystem


class RevolutionaryInterface:
    """الواجهة التفاعلية الثورية"""
    
    def __init__(self):
        """تهيئة الواجهة التفاعلية"""
        self.system = None
        self.running = True
        
    def start(self):
        """بدء الواجهة التفاعلية"""
        self.show_welcome()
        
        try:
            print("🔧 تهيئة النظام الثوري...")
            self.system = RevolutionaryShapeRecognitionSystem()
            print("✅ تم تهيئة النظام بنجاح!")
            
            self.main_menu()
            
        except Exception as e:
            print(f"❌ خطأ في تهيئة النظام: {e}")
            return
    
    def show_welcome(self):
        """عرض رسالة الترحيب"""
        print("\n" + "🌟" * 50)
        print("🚀 مرحباً بك في النظام الثوري للتعرف على الأشكال 🚀")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" * 50)
        print("\n💡 النظام الثوري يطبق مفهوم:")
        print("   📊 قاعدة بيانات + 🎨 وحدة رسم + 🔍 وحدة استنباط + 🧠 تعرف ذكي")
        print("   📏 مع السماحية والمسافة الإقليدية")
        print("   📝 ووصف ذكي: 'قطة بيضاء نائمة بخلفية بيوت وأشجار'")
    
    def main_menu(self):
        """القائمة الرئيسية"""
        while self.running:
            print("\n" + "="*60)
            print("📋 القائمة الرئيسية للنظام الثوري")
            print("="*60)
            print("1. 🎯 عرض المفهوم الثوري")
            print("2. 📊 عرض قاعدة البيانات")
            print("3. 🎨 اختبار وحدة الرسم")
            print("4. 🔍 اختبار وحدة الاستنباط")
            print("5. 🧠 اختبار التعرف الثوري")
            print("6. ➕ إضافة شكل جديد")
            print("7. 📈 إحصائيات النظام")
            print("8. 🔧 حالة النظام")
            print("9. 💾 حفظ النتائج")
            print("0. 🚪 خروج")
            print("="*60)
            
            choice = input("🎯 اختر رقم العملية: ").strip()
            
            if choice == "1":
                self.demonstrate_concept()
            elif choice == "2":
                self.show_database()
            elif choice == "3":
                self.test_drawing()
            elif choice == "4":
                self.test_extraction()
            elif choice == "5":
                self.test_recognition()
            elif choice == "6":
                self.add_new_shape()
            elif choice == "7":
                self.show_statistics()
            elif choice == "8":
                self.show_system_status()
            elif choice == "9":
                self.save_results()
            elif choice == "0":
                self.exit_system()
            else:
                print("❌ اختيار غير صحيح! حاول مرة أخرى.")
    
    def demonstrate_concept(self):
        """عرض المفهوم الثوري"""
        print("\n🎯 عرض المفهوم الثوري لباسل يحيى عبدالله...")
        self.system.demonstrate_revolutionary_concept()
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def show_database(self):
        """عرض قاعدة البيانات"""
        print("\n📊 قاعدة البيانات الثورية:")
        shapes = self.system.shape_db.get_all_shapes()
        stats = self.system.shape_db.get_statistics()
        
        print(f"📈 إجمالي الأشكال: {stats['total_shapes']}")
        print(f"📂 الفئات: {list(stats['categories'].keys())}")
        print(f"🎯 متوسط السماحية: {stats['average_tolerance']:.3f}")
        
        print("\n📋 قائمة الأشكال:")
        for i, shape in enumerate(shapes, 1):
            print(f"{i}. {shape.name} ({shape.category})")
            print(f"   🎨 اللون: {shape.color_properties['dominant_color']}")
            print(f"   📐 المساحة: {shape.geometric_features['area']}")
            print(f"   🎯 السماحية الإقليدية: {shape.tolerance_thresholds['euclidean_distance']}")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def test_drawing(self):
        """اختبار وحدة الرسم"""
        print("\n🎨 اختبار وحدة الرسم والتحريك...")
        shapes = self.system.shape_db.get_all_shapes()
        
        print("📋 الأشكال المتاحة:")
        for i, shape in enumerate(shapes, 1):
            print(f"{i}. {shape.name}")
        
        try:
            choice = int(input("🎯 اختر رقم الشكل للرسم: ")) - 1
            if 0 <= choice < len(shapes):
                selected_shape = shapes[choice]
                print(f"🖌️ رسم {selected_shape.name}...")
                
                result = self.system.drawing_unit.draw_shape_from_equation(selected_shape)
                print(f"📊 النتيجة: {result['success']}")
                print(f"🔧 الطريقة: {result['method']}")
                print(f"📝 الرسالة: {result['message']}")
                
                # حفظ الصورة إذا نجح الرسم
                if result["success"]:
                    save = input("💾 هل تريد حفظ الصورة؟ (y/n): ").lower()
                    if save == 'y':
                        filename = f"{selected_shape.name.replace(' ', '_')}.png"
                        success = self.system.drawing_unit.save_shape_image(selected_shape, filename)
                        if success:
                            print(f"✅ تم حفظ الصورة: {filename}")
            else:
                print("❌ اختيار غير صحيح!")
                
        except ValueError:
            print("❌ يرجى إدخال رقم صحيح!")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def test_extraction(self):
        """اختبار وحدة الاستنباط"""
        print("\n🔍 اختبار وحدة الاستنباط...")
        
        # إنشاء صورة اختبار
        print("🔧 إنشاء صورة اختبار...")
        test_image = self.create_test_image()
        
        print("🔍 استنباط الخصائص من الصورة...")
        result = self.system.extractor_unit.extract_equation_from_image(test_image)
        
        print(f"📊 النتيجة: {result['success']}")
        print(f"🔧 الطريقة: {result['method']}")
        print(f"📝 الرسالة: {result['message']}")
        
        if result["success"]:
            features = result["result"]
            print("\n📋 الخصائص المستنبطة:")
            print(f"🎨 اللون المهيمن: {features['color_properties']['dominant_color']}")
            print(f"📐 المساحة: {features['geometric_features']['area']:.1f}")
            print(f"🔄 الاستدارة: {features['geometric_features']['roundness']:.3f}")
            print(f"📏 نسبة العرض/الارتفاع: {features['geometric_features']['aspect_ratio']:.2f}")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def test_recognition(self):
        """اختبار التعرف الثوري"""
        print("\n🧠 اختبار التعرف الثوري...")
        
        # إنشاء صورة اختبار
        print("🔧 إنشاء صورة اختبار...")
        test_image = self.create_test_image()
        
        print("🔍 بدء التعرف الثوري...")
        result = self.system.recognition_engine.recognize_image(test_image)
        
        print(f"\n📊 نتيجة التعرف: {result['status']}")
        
        if result['status'] == "تم التعرف بنجاح":
            print(f"🎯 الشكل المتعرف عليه: {result['recognized_shape']}")
            print(f"📂 الفئة: {result['category']}")
            print(f"📈 مستوى الثقة: {result['confidence']:.2%}")
            print(f"📏 المسافة الإقليدية: {result['euclidean_distance']:.4f}")
            print(f"📐 التشابه الهندسي: {result['geometric_similarity']:.4f}")
            print(f"🌈 التشابه اللوني: {result['color_similarity']:.4f}")
            print(f"📝 الوصف الذكي: {result['description']}")
        else:
            print(f"❌ {result['message']}")
            if 'closest_match' in result:
                print(f"🔍 أقرب تطابق: {result['closest_match']}")
                print(f"📏 المسافة: {result['euclidean_distance']:.4f}")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def add_new_shape(self):
        """إضافة شكل جديد"""
        print("\n➕ إضافة شكل جديد للنظام...")
        
        name = input("📝 اسم الشكل الجديد: ").strip()
        if not name:
            print("❌ يجب إدخال اسم للشكل!")
            return
        
        print("📂 الفئات المتاحة: حيوانات، مباني، نباتات، أخرى")
        category = input("📂 فئة الشكل: ").strip()
        if not category:
            category = "أخرى"
        
        # إنشاء صورة اختبار للشكل الجديد
        print("🔧 إنشاء صورة للشكل الجديد...")
        test_image = self.create_test_image()
        
        # إضافة الشكل
        success = self.system.add_new_shape(name, category, test_image)
        
        if success:
            print(f"✅ تم إضافة الشكل الجديد: {name}")
        else:
            print(f"❌ فشل في إضافة الشكل: {name}")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def show_statistics(self):
        """عرض إحصائيات النظام"""
        print("\n📈 إحصائيات النظام الثوري:")
        
        # إحصائيات قاعدة البيانات
        db_stats = self.system.shape_db.get_statistics()
        print(f"📊 قاعدة البيانات:")
        print(f"   📈 إجمالي الأشكال: {db_stats['total_shapes']}")
        print(f"   📂 عدد الفئات: {len(db_stats['categories'])}")
        print(f"   🎯 متوسط السماحية: {db_stats['average_tolerance']:.3f}")
        
        # إحصائيات التعرف
        recognition_stats = self.system.recognition_engine.get_recognition_statistics()
        print(f"\n🧠 إحصائيات التعرف:")
        print(f"   🔍 إجمالي عمليات التعرف: {recognition_stats.get('total_recognitions', 0)}")
        print(f"   📈 متوسط الثقة: {recognition_stats.get('average_confidence', 0):.2%}")
        
        if recognition_stats.get('top_recognized_shapes'):
            print(f"   🏆 الأشكال الأكثر تعرفاً:")
            for shape_name, count in recognition_stats['top_recognized_shapes']:
                print(f"      • {shape_name}: {count} مرة")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def show_system_status(self):
        """عرض حالة النظام"""
        print("\n🔧 حالة النظام الثوري:")
        status = self.system.get_system_status()
        
        print(f"🌟 النظام: {status['system_name']}")
        print(f"👨‍💻 المبدع: {status['creator']}")
        print(f"📦 الإصدار: {status['version']}")
        
        print(f"\n🔧 المكونات:")
        for component, details in status['components'].items():
            print(f"   • {component}: {details}")
        
        print(f"\n💡 الميزات الثورية:")
        for feature in status['revolutionary_features']:
            print(f"   ✅ {feature}")
        
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def save_results(self):
        """حفظ النتائج"""
        print("\n💾 حفظ نتائج النظام...")
        
        # إنشاء صورة اختبار ومعالجتها
        test_image = self.create_test_image()
        result = self.system.process_image(test_image, save_results=True)
        
        print("✅ تم حفظ النتائج بنجاح!")
        input("\n⏸️ اضغط Enter للمتابعة...")
    
    def create_test_image(self) -> np.ndarray:
        """إنشاء صورة اختبار بسيطة"""
        # إنشاء صورة بسيطة للاختبار
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # رسم دائرة بيضاء
        center = (100, 100)
        radius = 50
        y, x = np.ogrid[:200, :200]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = [255, 255, 255]
        
        return image
    
    def exit_system(self):
        """خروج من النظام"""
        print("\n🚪 خروج من النظام الثوري...")
        print("🌟 شكراً لاستخدام النظام الثوري لباسل يحيى عبدالله!")
        print("🌟 نظام بصيرة - العراق/الموصل 🌟")
        self.running = False


def main():
    """الدالة الرئيسية للواجهة التفاعلية"""
    interface = RevolutionaryInterface()
    interface.start()


if __name__ == "__main__":
    main()
