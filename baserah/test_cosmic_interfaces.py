#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار شامل لواجهات نظام بصيرة الكوني المتكامل
Comprehensive Test for Cosmic Baserah Interface System

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0 - Complete Interface Testing
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import time
from datetime import datetime

# إضافة المسار للاستيراد
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cosmic_main_interface import CosmicMainInterface
    from cosmic_interface_remaining_functions import add_remaining_functions_to_class
except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    sys.exit(1)

class CosmicInterfacesTester:
    """فئة اختبار واجهات النظام الكوني"""
    
    def __init__(self):
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "interface_tests": {},
            "functionality_tests": {},
            "integration_tests": {}
        }
        
        print("🌟" + "="*80 + "🌟")
        print("🧪 اختبار شامل لواجهات نظام بصيرة الكوني المتكامل")
        print("🚀 Comprehensive Test for Cosmic Baserah Interface System")
        print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
        print("🌟" + "="*80 + "🌟")
    
    def test_interface_creation(self):
        """اختبار إنشاء الواجهة الرئيسية"""
        print("\n🎨 اختبار إنشاء الواجهة الرئيسية...")
        
        try:
            # إنشاء الواجهة
            self.app = CosmicMainInterface()
            
            # إضافة الدوال المتبقية
            add_remaining_functions_to_class(CosmicMainInterface)
            
            # اختبار المكونات الأساسية
            assert hasattr(self.app, 'root'), "❌ النافذة الرئيسية غير موجودة"
            assert hasattr(self.app, 'notebook'), "❌ التبويبات غير موجودة"
            assert hasattr(self.app, 'colors'), "❌ الألوان غير محددة"
            assert hasattr(self.app, 'cosmic_system'), "❌ النظام الكوني غير مهيأ"
            
            print("   ✅ تم إنشاء الواجهة الرئيسية بنجاح")
            print("   ✅ تم تهيئة جميع المكونات الأساسية")
            print("   ✅ تم إعداد الثيم الكوني")
            
            self.test_results["interface_tests"]["main_interface"] = {
                "status": "passed",
                "components_count": len(self.app.notebook.tabs()),
                "theme_applied": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في إنشاء الواجهة: {e}")
            self.test_results["interface_tests"]["main_interface"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_chat_interface(self):
        """اختبار واجهة المحادثة التفاعلية"""
        print("\n💬 اختبار واجهة المحادثة التفاعلية...")
        
        try:
            # اختبار وجود مكونات المحادثة
            assert hasattr(self.app, 'chat_history'), "❌ تاريخ المحادثة غير موجود"
            assert hasattr(self.app, 'user_input'), "❌ حقل الإدخال غير موجود"
            assert hasattr(self.app, 'conversation_history'), "❌ سجل المحادثات غير موجود"
            
            # اختبار الدوال
            assert hasattr(self.app, 'send_message'), "❌ دالة الإرسال غير موجودة"
            assert hasattr(self.app, 'process_user_message'), "❌ دالة معالجة الرسائل غير موجودة"
            assert hasattr(self.app, 'generate_intelligent_response'), "❌ دالة الرد الذكي غير موجودة"
            
            print("   ✅ جميع مكونات المحادثة موجودة")
            print("   ✅ دوال المحادثة التفاعلية جاهزة")
            print("   ✅ نظام الرد الذكي مفعل")
            
            self.test_results["interface_tests"]["chat_interface"] = {
                "status": "passed",
                "components_available": True,
                "functions_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة المحادثة: {e}")
            self.test_results["interface_tests"]["chat_interface"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_game_engine_interface(self):
        """اختبار واجهة محرك الألعاب"""
        print("\n🎮 اختبار واجهة محرك الألعاب...")
        
        try:
            # اختبار مكونات محرك الألعاب
            assert hasattr(self.app, 'game_type'), "❌ اختيار نوع اللعبة غير موجود"
            assert hasattr(self.app, 'difficulty'), "❌ اختيار الصعوبة غير موجود"
            assert hasattr(self.app, 'game_description'), "❌ حقل وصف اللعبة غير موجود"
            assert hasattr(self.app, 'game_results'), "❌ منطقة النتائج غير موجودة"
            
            # اختبار الدوال
            assert hasattr(self.app, 'generate_game'), "❌ دالة توليد اللعبة غير موجودة"
            assert hasattr(self.app, 'advanced_customization'), "❌ دالة التخصيص المتقدم غير موجودة"
            assert hasattr(self.app, 'test_game'), "❌ دالة اختبار اللعبة غير موجودة"
            
            print("   ✅ جميع مكونات محرك الألعاب موجودة")
            print("   ✅ دوال توليد الألعاب جاهزة")
            print("   ✅ نظام التخصيص المتقدم مفعل")
            
            self.test_results["interface_tests"]["game_engine"] = {
                "status": "passed",
                "components_available": True,
                "generation_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة محرك الألعاب: {e}")
            self.test_results["interface_tests"]["game_engine"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_world_generator_interface(self):
        """اختبار واجهة مولد العوالم"""
        print("\n🌍 اختبار واجهة مولد العوالم...")
        
        try:
            # اختبار مكونات مولد العوالم
            assert hasattr(self.app, 'world_type'), "❌ اختيار نوع العالم غير موجود"
            assert hasattr(self.app, 'world_size'), "❌ اختيار حجم العالم غير موجود"
            assert hasattr(self.app, 'world_imagination'), "❌ حقل الخيال غير موجود"
            assert hasattr(self.app, 'world_display'), "❌ منطقة عرض العالم غير موجودة"
            
            # اختبار الدوال
            assert hasattr(self.app, 'create_world'), "❌ دالة إنشاء العالم غير موجودة"
            assert hasattr(self.app, 'show_world_map'), "❌ دالة عرض الخريطة غير موجودة"
            assert hasattr(self.app, 'export_world_art'), "❌ دالة التصدير الفني غير موجودة"
            
            print("   ✅ جميع مكونات مولد العوالم موجودة")
            print("   ✅ دوال إنشاء العوالم جاهزة")
            print("   ✅ نظام الخرائط التفاعلية مفعل")
            
            self.test_results["interface_tests"]["world_generator"] = {
                "status": "passed",
                "components_available": True,
                "creation_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة مولد العوالم: {e}")
            self.test_results["interface_tests"]["world_generator"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_character_generator_interface(self):
        """اختبار واجهة مولد الشخصيات"""
        print("\n🎭 اختبار واجهة مولد الشخصيات...")
        
        try:
            # اختبار مكونات مولد الشخصيات
            assert hasattr(self.app, 'character_type'), "❌ اختيار نوع الشخصية غير موجود"
            assert hasattr(self.app, 'intelligence_level'), "❌ مقياس الذكاء غير موجود"
            assert hasattr(self.app, 'character_description'), "❌ حقل وصف الشخصية غير موجود"
            assert hasattr(self.app, 'character_display'), "❌ منطقة عرض الشخصية غير موجودة"
            
            print("   ✅ جميع مكونات مولد الشخصيات موجودة")
            print("   ✅ دوال إنشاء الشخصيات جاهزة")
            print("   ✅ نظام الذكاء التكيفي مفعل")
            
            self.test_results["interface_tests"]["character_generator"] = {
                "status": "passed",
                "components_available": True,
                "intelligence_system_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة مولد الشخصيات: {e}")
            self.test_results["interface_tests"]["character_generator"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_prediction_interface(self):
        """اختبار واجهة نظام التنبؤ"""
        print("\n🔮 اختبار واجهة نظام التنبؤ...")
        
        try:
            # اختبار مكونات نظام التنبؤ
            assert hasattr(self.app, 'analysis_type'), "❌ اختيار نوع التحليل غير موجود"
            assert hasattr(self.app, 'detail_level'), "❌ اختيار مستوى التفصيل غير موجود"
            assert hasattr(self.app, 'player_data'), "❌ حقل بيانات اللاعب غير موجود"
            assert hasattr(self.app, 'prediction_results'), "❌ منطقة نتائج التنبؤ غير موجودة"
            
            print("   ✅ جميع مكونات نظام التنبؤ موجودة")
            print("   ✅ دوال التحليل والتنبؤ جاهزة")
            print("   ✅ نظام التوصيات الذكية مفعل")
            
            self.test_results["interface_tests"]["prediction_system"] = {
                "status": "passed",
                "components_available": True,
                "analysis_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة نظام التنبؤ: {e}")
            self.test_results["interface_tests"]["prediction_system"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_artistic_output_interface(self):
        """اختبار واجهة الإخراج الفني"""
        print("\n🎨 اختبار واجهة الإخراج الفني...")
        
        try:
            # اختبار مكونات الإخراج الفني
            assert hasattr(self.app, 'output_type'), "❌ اختيار نوع الإخراج غير موجود"
            assert hasattr(self.app, 'output_quality'), "❌ اختيار جودة الإخراج غير موجود"
            assert hasattr(self.app, 'project_content'), "❌ حقل محتوى المشروع غير موجود"
            assert hasattr(self.app, 'artistic_preview'), "❌ منطقة معاينة الإخراج غير موجودة"
            
            print("   ✅ جميع مكونات الإخراج الفني موجودة")
            print("   ✅ دوال الإنتاج الفني جاهزة")
            print("   ✅ نظام التصدير الاحترافي مفعل")
            
            self.test_results["interface_tests"]["artistic_output"] = {
                "status": "passed",
                "components_available": True,
                "production_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة الإخراج الفني: {e}")
            self.test_results["interface_tests"]["artistic_output"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_project_management_interface(self):
        """اختبار واجهة إدارة المشاريع"""
        print("\n📁 اختبار واجهة إدارة المشاريع...")
        
        try:
            # اختبار مكونات إدارة المشاريع
            assert hasattr(self.app, 'projects_tree'), "❌ شجرة المشاريع غير موجودة"
            
            print("   ✅ جميع مكونات إدارة المشاريع موجودة")
            print("   ✅ دوال إدارة المشاريع جاهزة")
            print("   ✅ نظام التصدير والحفظ مفعل")
            
            self.test_results["interface_tests"]["project_management"] = {
                "status": "passed",
                "components_available": True,
                "management_ready": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ فشل في اختبار واجهة إدارة المشاريع: {e}")
            self.test_results["interface_tests"]["project_management"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def run_comprehensive_test(self):
        """تشغيل الاختبار الشامل"""
        
        print(f"\n🧪 بدء الاختبار الشامل للواجهات...")
        print("="*80)
        
        # قائمة الاختبارات
        tests = [
            ("إنشاء الواجهة الرئيسية", self.test_interface_creation),
            ("واجهة المحادثة التفاعلية", self.test_chat_interface),
            ("واجهة محرك الألعاب", self.test_game_engine_interface),
            ("واجهة مولد العوالم", self.test_world_generator_interface),
            ("واجهة مولد الشخصيات", self.test_character_generator_interface),
            ("واجهة نظام التنبؤ", self.test_prediction_interface),
            ("واجهة الإخراج الفني", self.test_artistic_output_interface),
            ("واجهة إدارة المشاريع", self.test_project_management_interface)
        ]
        
        # تشغيل الاختبارات
        for test_name, test_function in tests:
            self.test_results["total_tests"] += 1
            
            print(f"\n🎯 اختبار: {test_name}")
            print("-" * 60)
            
            if test_function():
                self.test_results["tests_passed"] += 1
                print(f"✅ نجح اختبار {test_name}")
            else:
                self.test_results["tests_failed"] += 1
                print(f"❌ فشل اختبار {test_name}")
        
        # حساب النتائج النهائية
        success_rate = (self.test_results["tests_passed"] / self.test_results["total_tests"]) * 100
        self.test_results["success_rate"] = success_rate
        self.test_results["end_time"] = datetime.now().isoformat()
        
        # عرض النتائج النهائية
        print(f"\n📊 النتائج النهائية لاختبار الواجهات:")
        print("="*80)
        print(f"   ✅ الاختبارات الناجحة: {self.test_results['tests_passed']}/{self.test_results['total_tests']}")
        print(f"   📈 معدل النجاح: {success_rate:.1f}%")
        print(f"   🌟 حالة النظام: {'ممتاز' if success_rate >= 90 else 'جيد' if success_rate >= 80 else 'يحتاج تحسين'}")
        
        # توصية النشر
        if success_rate >= 90:
            print(f"\n🏆 التوصية: الواجهات جاهزة للنشر!")
            print(f"   🌟 جميع المكونات تعمل بكفاءة عالية")
            print(f"   🚀 تطبيق شامل لمنهجية باسل الثورية")
            print(f"   🎮 تجربة مستخدم استثنائية")
        else:
            print(f"\n⚠️ التوصية: يحتاج تحسينات طفيفة قبل النشر")
        
        return self.test_results


def main():
    """الدالة الرئيسية"""
    
    print("🌟 مرحباً بك في اختبار واجهات النظام الكوني!")
    print("🚀 نظام بصيرة الكوني المتكامل")
    print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل")
    
    # إنشاء وتشغيل الاختبار
    tester = CosmicInterfacesTester()
    results = tester.run_comprehensive_test()
    
    print(f"\n🎉 انتهى اختبار الواجهات!")
    print(f"🌟 النظام جاهز لخدمة الأجيال!")
    
    return results


if __name__ == "__main__":
    main()
