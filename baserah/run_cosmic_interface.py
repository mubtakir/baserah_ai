#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تشغيل واجهة نظام بصيرة الكوني المتكامل
Run Cosmic Baserah Integrated System Interface

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0 - Complete Interface Runner
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
from datetime import datetime

# إضافة المسار للاستيراد
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_requirements():
    """فحص المتطلبات الأساسية"""
    print("🔍 فحص المتطلبات الأساسية...")
    
    required_modules = ['tkinter', 'json', 'datetime', 'threading', 'time']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"   ❌ {module}")
    
    if missing_modules:
        print(f"\n❌ المتطلبات المفقودة: {', '.join(missing_modules)}")
        return False
    
    print("✅ جميع المتطلبات متوفرة")
    return True

def display_welcome_message():
    """عرض رسالة الترحيب"""
    welcome_msg = """
🌟════════════════════════════════════════════════════════════════════════════════🌟
                                                                                    
    🌟 نظام بصيرة الكوني المتكامل 🌟
    Cosmic Baserah Integrated System
                                                                                    
    🚀 النظام الثوري لمستقبل صناعة الألعاب 🚀
    Revolutionary System for the Future of Game Development
                                                                                    
    🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟
    Created by Basil Yahya Abdullah from Iraq/Mosul
                                                                                    
🌟════════════════════════════════════════════════════════════════════════════════🌟

🎯 الميزات الرئيسية:
   🎮 محرك الألعاب الكوني - توليد ألعاب مبتكرة من الأفكار
   🌍 مولد العوالم الذكي - إنشاء عوالم خيالية مذهلة
   🎭 مولد الشخصيات الذكي - تطوير شخصيات تفاعلية معقدة
   🔮 نظام التنبؤ الذكي - فهم وتوقع سلوك اللاعبين
   🎨 الإخراج الفني الاحترافي - تحويل الأفكار إلى محتوى فني
   💬 المحادثة التفاعلية - حوار ذكي مع النظام
   📁 إدارة المشاريع - تنظيم وحفظ جميع الأعمال

🌟 منهجية باسل الثورية:
   🧠 التفكير التكاملي - دمج جميع العناصر في نظام متناغم
   🚀 الإبداع الثوري - كسر الحدود التقليدية
   💡 الحكمة التطبيقية - تطبيق المعرفة بحكمة
   🔄 التكيف الذكي - التطور المستمر مع البيئة

🎯 الهدف: تحويل صناعة الألعاب إلى منصة للنمو الإنساني والإبداع الكوني

🚀 جاهز للانطلاق نحو المستقبل!
"""
    print(welcome_msg)

def run_interface():
    """تشغيل الواجهة الرئيسية"""
    try:
        print("🚀 تشغيل الواجهة الرئيسية...")
        
        # استيراد الواجهة
        from cosmic_main_interface import CosmicMainInterface
        from cosmic_interface_remaining_functions import add_remaining_functions_to_class
        
        print("✅ تم تحميل مكونات الواجهة")
        
        # إضافة الدوال المتبقية
        add_remaining_functions_to_class(CosmicMainInterface)
        print("✅ تم دمج جميع الدوال الوظيفية")
        
        # إنشاء وتشغيل الواجهة
        app = CosmicMainInterface()
        
        # إضافة الشريط السفلي
        app.create_footer()
        
        print("✅ تم إنشاء الواجهة بنجاح")
        print("🌟 الواجهة جاهزة للاستخدام!")
        print("\n" + "="*80)
        print("🎉 مرحباً بك في نظام بصيرة الكوني!")
        print("💡 استخدم تبويب 'المحادثة التفاعلية' للبدء")
        print("🌟 استمتع بتجربة الإبداع الكوني!")
        print("="*80)
        
        # تشغيل الواجهة
        app.run()
        
    except ImportError as e:
        error_msg = f"""
❌ خطأ في تحميل الواجهة: {e}

🔧 الحلول المقترحة:
1. تأكد من وجود ملف cosmic_main_interface.py
2. تأكد من وجود ملف cosmic_interface_remaining_functions.py
3. تأكد من أن جميع الملفات في نفس المجلد
4. تحقق من صحة بناء الملفات

📞 للمساعدة: راجع دليل المطورين أو اتصل بالدعم التقني
"""
        print(error_msg)
        return False
        
    except Exception as e:
        error_msg = f"""
❌ خطأ غير متوقع: {e}

🔧 الحلول المقترحة:
1. أعد تشغيل البرنامج
2. تحقق من متطلبات النظام
3. تأكد من صلاحيات الوصول للملفات

📞 للمساعدة: راجع دليل استكشاف الأخطاء
"""
        print(error_msg)
        return False

def show_system_info():
    """عرض معلومات النظام"""
    info = f"""
📊 معلومات النظام:
   🐍 Python: {sys.version.split()[0]}
   💻 النظام: {os.name}
   📅 التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   📁 المجلد: {os.getcwd()}

🌟 معلومات المشروع:
   📛 الاسم: نظام بصيرة الكوني المتكامل
   🔢 الإصدار: 1.0.0
   👨‍💻 المطور: باسل يحيى عبدالله
   🌍 المكان: العراق/الموصل
   📧 الدعم: cosmic-baserah-support@example.com
"""
    print(info)

def main():
    """الدالة الرئيسية"""
    
    # عرض رسالة الترحيب
    display_welcome_message()
    
    # عرض معلومات النظام
    show_system_info()
    
    # فحص المتطلبات
    if not check_requirements():
        print("\n❌ لا يمكن تشغيل النظام بسبب نقص المتطلبات")
        input("اضغط Enter للخروج...")
        return
    
    # تشغيل الواجهة
    print("\n🚀 بدء تشغيل النظام...")
    print("⏳ يرجى الانتظار...")
    
    try:
        run_interface()
    except KeyboardInterrupt:
        print("\n\n🛑 تم إيقاف النظام بواسطة المستخدم")
        print("🌟 شكراً لاستخدام نظام بصيرة الكوني!")
    except Exception as e:
        print(f"\n❌ خطأ في تشغيل النظام: {e}")
    finally:
        print("\n🌟 إلى اللقاء! نتطلع لرؤيتك مرة أخرى")
        print("🚀 نظام بصيرة الكوني - حيث تلتقي الحكمة بالتقنية")

if __name__ == "__main__":
    main()
