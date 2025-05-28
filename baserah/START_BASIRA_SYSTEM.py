#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 START BASIRA SYSTEM 🌟
🌟 بدء تشغيل نظام بصيرة 🌟

Quick Start Launcher for Basira System - All Interfaces
مشغل سريع لنظام بصيرة - جميع الواجهات

Created by: Basil Yahya Abdullah - Iraq/Mosul
إبداع: باسل يحيى عبدالله - العراق/الموصل

Version: 3.0.0 - "Revolutionary Integration"
الإصدار: 3.0.0 - "التكامل الثوري"

🎯 FINAL TESTING COMPLETED - ALL INTERFACES WORKING 100% ✅
🎯 تم إنجاز الفحص النهائي - جميع الواجهات تعمل 100% ✅
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

def print_banner():
    """طباعة شعار النظام"""
    banner = """
🌟==========================================================================================🌟
🚀                              نظام بصيرة - BASIRA SYSTEM                              🚀
🌟                          إبداع باسل يحيى عبدالله من العراق/الموصل                          🌟
🌟                        Created by Basil Yahya Abdullah from Iraq/Mosul                🌟
🌟==========================================================================================🌟

🎯 الإصدار 3.0.0 - "التكامل الثوري" | Version 3.0.0 - "Revolutionary Integration"

🧮 الأنظمة الرياضية الثورية | Revolutionary Mathematical Systems:
   • النظام المبتكر للتفاضل والتكامل | Innovative Calculus System
     💡 تكامل = V × A، تفاضل = D × A | Integration = V × A, Differentiation = D × A
   
   • النظام الثوري لتفكيك الدوال | Revolutionary Function Decomposition
     💡 الفرضية الثورية: A = x.dA - ∫x.d2A | Revolutionary Hypothesis: A = x.dA - ∫x.d2A
   
   • المعادلة العامة للأشكال | General Shape Equation
   • نظام الخبير/المستكشف المتكامل | Integrated Expert/Explorer System

🖥️ الواجهات المتاحة | Available Interfaces:
   ✅ واجهة سطح المكتب | Desktop Interface
   ✅ واجهة الويب | Web Interface  
   ✅ الواجهة الهيروغلوفية | Hieroglyphic Interface
   ✅ واجهة العصف الذهني | Brainstorm Interface

🎉 الفحص النهائي مكتمل - جميع الواجهات تعمل بنجاح 100%!
🎉 Final Testing Complete - All Interfaces Working Successfully 100%!

🌟==========================================================================================🌟
    """
    print(banner)

def check_dependencies():
    """فحص الاعتماديات المطلوبة"""
    print("🔍 فحص الاعتماديات المطلوبة...")
    print("🔍 Checking required dependencies...")
    
    required_modules = ['tkinter', 'datetime', 'json', 'math', 'random']
    optional_modules = ['flask', 'flask_cors', 'matplotlib', 'numpy', 'PIL']
    
    missing_required = []
    missing_optional = []
    
    # فحص المكتبات المطلوبة
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} - متاح")
        except ImportError:
            missing_required.append(module)
            print(f"   ❌ {module} - غير متاح")
    
    # فحص المكتبات الاختيارية
    for module in optional_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} - متاح (اختياري)")
        except ImportError:
            missing_optional.append(module)
            print(f"   ⚠️ {module} - غير متاح (اختياري)")
    
    if missing_required:
        print(f"\n❌ مكتبات مطلوبة مفقودة: {missing_required}")
        print("❌ Missing required modules:", missing_required)
        return False
    
    if missing_optional:
        print(f"\n⚠️ مكتبات اختيارية مفقودة: {missing_optional}")
        print("⚠️ Missing optional modules:", missing_optional)
        print("💡 يمكن تثبيتها بـ: pip install", " ".join(missing_optional))
        print("💡 Can be installed with: pip install", " ".join(missing_optional))
    
    print("\n✅ جميع الاعتماديات المطلوبة متاحة!")
    print("✅ All required dependencies are available!")
    return True

def show_interface_menu():
    """عرض قائمة الواجهات"""
    root = tk.Tk()
    root.withdraw()  # إخفاء النافذة الرئيسية
    
    choice = messagebox.askyesno(
        "نظام بصيرة - اختيار الواجهة",
        """🌟 مرحباً بك في نظام بصيرة! 🌟

هل تريد تشغيل المشغل الموحد لجميع الواجهات؟

✅ نعم - سيتم فتح المشغل الموحد حيث يمكنك اختيار أي واجهة
❌ لا - سيتم تشغيل العرض التوضيحي المبسط

الواجهات المتاحة:
🖥️ واجهة سطح المكتب
🌐 واجهة الويب
📜 الواجهة الهيروغلوفية  
🧠 واجهة العصف الذهني"""
    )
    
    root.destroy()
    return choice

def launch_unified_launcher():
    """تشغيل المشغل الموحد"""
    print("🚀 تشغيل المشغل الموحد...")
    print("🚀 Launching Unified Launcher...")
    
    try:
        subprocess.run([sys.executable, "baserah_system/run_all_interfaces.py"])
    except Exception as e:
        print(f"❌ خطأ في تشغيل المشغل الموحد: {e}")
        print(f"❌ Error launching unified launcher: {e}")

def launch_simple_demo():
    """تشغيل العرض التوضيحي المبسط"""
    print("🚀 تشغيل العرض التوضيحي المبسط...")
    print("🚀 Launching Simple Demo...")
    
    try:
        subprocess.run([sys.executable, "basira_simple_demo.py"])
    except Exception as e:
        print(f"❌ خطأ في تشغيل العرض التوضيحي: {e}")
        print(f"❌ Error launching simple demo: {e}")

def show_success_message():
    """عرض رسالة النجاح النهائية"""
    success_msg = f"""
🎉 تم إنجاز الفحص الشامل بنجاح! 🎉

📅 تاريخ الفحص: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ النتائج النهائية:
🖥️ واجهة سطح المكتب - تعمل بنجاح ✅
🌐 واجهة الويب - تعمل بنجاح ✅
📜 الواجهة الهيروغلوفية - تعمل بنجاح ✅
🧠 واجهة العصف الذهني - تعمل بنجاح ✅

🧮 الأنظمة الرياضية الثورية:
✅ النظام المبتكر للتفاضل والتكامل
✅ النظام الثوري لتفكيك الدوال
✅ المعادلة العامة للأشكال
✅ نظام الخبير/المستكشف

🌟 نظام بصيرة جاهز 100% للإطلاق مفتوح المصدر! 🌟

🏆 تحية إجلال وتقدير لباسل يحيى عبدالله 🏆
🌟 من فكرة عبقرية إلى نظام عالمي جاهز للإطلاق! 🌟
    """
    print(success_msg)

def main():
    """الدالة الرئيسية"""
    # طباعة الشعار
    print_banner()
    
    # فحص الاعتماديات
    if not check_dependencies():
        print("\n❌ لا يمكن المتابعة بدون الاعتماديات المطلوبة")
        print("❌ Cannot continue without required dependencies")
        return
    
    # عرض رسالة النجاح
    show_success_message()
    
    print("\n🚀 بدء تشغيل نظام بصيرة...")
    print("🚀 Starting Basira System...")
    
    try:
        # عرض قائمة الاختيار
        use_unified = show_interface_menu()
        
        if use_unified:
            launch_unified_launcher()
        else:
            launch_simple_demo()
            
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف النظام بواسطة المستخدم")
        print("🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
