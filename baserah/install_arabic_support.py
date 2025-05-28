#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install Arabic Text Support for Basira System
تثبيت دعم النصوص العربية لنظام بصيرة

This script installs the required libraries for proper Arabic text display
in Python GUI applications.

يقوم هذا الملف بتثبيت المكتبات المطلوبة لعرض النصوص العربية بشكل صحيح
في تطبيقات Python GUI.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import subprocess
import sys
import os
from datetime import datetime

def print_banner():
    """طباعة شعار التثبيت"""
    banner = """
🌟==========================================================================================🌟
🔤                          تثبيت دعم النصوص العربية لنظام بصيرة                          🔤
🔤                        Install Arabic Text Support for Basira System                  🔤
🌟                          إبداع باسل يحيى عبدالله من العراق/الموصل                          🌟
🌟                        Created by Basil Yahya Abdullah from Iraq/Mosul                🌟
🌟==========================================================================================🌟

📅 التاريخ: {date}
🎯 الهدف: حل مشكلة اتجاه النصوص العربية في واجهات Python
🎯 Goal: Fix Arabic text direction issues in Python interfaces

🔧 المكتبات المطلوبة:
   • arabic-reshaper - لإعادة تشكيل النصوص العربية
   • python-bidi - لتطبيق خوارزمية BiDi (اتجاه النص)

🔧 Required Libraries:
   • arabic-reshaper - For reshaping Arabic text
   • python-bidi - For applying BiDi (text direction) algorithm

🌟==========================================================================================🌟
    """.format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(banner)

def check_python_version():
    """فحص إصدار Python"""
    print("🔍 فحص إصدار Python...")
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    print(f"📊 إصدار Python: {version.major}.{version.minor}.{version.micro}")
    print(f"📊 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("❌ يتطلب Python 3.6 أو أحدث")
        print("❌ Requires Python 3.6 or newer")
        return False
    
    print("✅ إصدار Python مناسب")
    print("✅ Python version is suitable")
    return True

def check_pip():
    """فحص توفر pip"""
    print("\n🔍 فحص توفر pip...")
    print("🔍 Checking pip availability...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip متاح")
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip غير متاح")
        print("❌ pip is not available")
        return False

def install_package(package_name, description_ar, description_en):
    """تثبيت مكتبة واحدة"""
    print(f"\n🔄 تثبيت {package_name}...")
    print(f"🔄 Installing {package_name}...")
    print(f"📝 الوصف: {description_ar}")
    print(f"📝 Description: {description_en}")
    
    try:
        # تثبيت المكتبة
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ تم تثبيت {package_name} بنجاح!")
        print(f"✅ {package_name} installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ فشل في تثبيت {package_name}")
        print(f"❌ Failed to install {package_name}")
        print(f"🔍 رسالة الخطأ: {e.stderr}")
        print(f"🔍 Error message: {e.stderr}")
        return False

def test_installation():
    """اختبار التثبيت"""
    print("\n🧪 اختبار التثبيت...")
    print("🧪 Testing installation...")
    
    try:
        # اختبار arabic-reshaper
        print("🔍 اختبار arabic-reshaper...")
        from arabic_reshaper import reshape
        test_text = "نظام بصيرة"
        reshaped = reshape(test_text)
        print(f"✅ arabic-reshaper يعمل: {test_text} -> {reshaped}")
        
        # اختبار python-bidi
        print("🔍 اختبار python-bidi...")
        from bidi.algorithm import get_display
        display_text = get_display(reshaped)
        print(f"✅ python-bidi يعمل: {reshaped} -> {display_text}")
        
        print("\n🎉 جميع المكتبات تعمل بنجاح!")
        print("🎉 All libraries are working successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ خطأ في الاستيراد: {e}")
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        print(f"❌ Test error: {e}")
        return False

def create_test_file():
    """إنشاء ملف اختبار للنصوص العربية"""
    print("\n📝 إنشاء ملف اختبار...")
    print("📝 Creating test file...")
    
    test_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Text Test - نظام بصيرة
اختبار النصوص العربية - نظام بصيرة
"""

import tkinter as tk
from arabic_reshaper import reshape
from bidi.algorithm import get_display

def fix_arabic_text(text):
    """إصلاح اتجاه النص العربي"""
    reshaped_text = reshape(text)
    display_text = get_display(reshaped_text)
    return display_text

def test_arabic_gui():
    """اختبار واجهة مع النصوص العربية"""
    root = tk.Tk()
    root.title("اختبار النصوص العربية")
    root.geometry("400x300")
    
    # نصوص للاختبار
    test_texts = [
        "نظام بصيرة",
        "إبداع باسل يحيى عبدالله",
        "النظام المبتكر للتفاضل والتكامل",
        "النظام الثوري لتفكيك الدوال"
    ]
    
    tk.Label(root, text="اختبار النصوص العربية", 
             font=('Arial', 16, 'bold')).pack(pady=10)
    
    for text in test_texts:
        # النص الأصلي
        tk.Label(root, text=f"الأصلي: {text}", 
                font=('Arial', 10)).pack(pady=2)
        
        # النص المُصحح
        fixed_text = fix_arabic_text(text)
        tk.Label(root, text=f"المُصحح: {fixed_text}", 
                font=('Arial', 10), fg='blue').pack(pady=2)
    
    tk.Button(root, text=fix_arabic_text("إغلاق"), 
              command=root.destroy).pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    test_arabic_gui()
'''
    
    try:
        with open("test_arabic_text.py", "w", encoding="utf-8") as f:
            f.write(test_code)
        
        print("✅ تم إنشاء ملف test_arabic_text.py")
        print("✅ Created test_arabic_text.py file")
        print("💡 يمكنك تشغيله بـ: python test_arabic_text.py")
        print("💡 You can run it with: python test_arabic_text.py")
        return True
        
    except Exception as e:
        print(f"❌ فشل في إنشاء ملف الاختبار: {e}")
        print(f"❌ Failed to create test file: {e}")
        return False

def show_usage_instructions():
    """عرض تعليمات الاستخدام"""
    instructions = """
🎯 تعليمات الاستخدام | Usage Instructions:

📚 في الكود الخاص بك، استخدم:
📚 In your code, use:

```python
from arabic_reshaper import reshape
from bidi.algorithm import get_display

def fix_arabic_text(text):
    reshaped_text = reshape(text)
    display_text = get_display(reshaped_text)
    return display_text

# استخدام في tkinter
label = tk.Label(root, text=fix_arabic_text("نظام بصيرة"))
```

🔧 أو استخدم معالج النصوص في نظام بصيرة:
🔧 Or use the text handler in Basira System:

```python
from baserah_system.arabic_text_handler import fix_arabic_text

label = tk.Label(root, text=fix_arabic_text("نظام بصيرة"))
```

🌟 الآن ستظهر النصوص العربية بالاتجاه الصحيح!
🌟 Now Arabic texts will display in the correct direction!
    """
    print(instructions)

def main():
    """الدالة الرئيسية"""
    print_banner()
    
    # فحص المتطلبات الأساسية
    if not check_python_version():
        return False
    
    if not check_pip():
        print("\n💡 يرجى تثبيت pip أولاً")
        print("💡 Please install pip first")
        return False
    
    # قائمة المكتبات للتثبيت
    packages = [
        {
            'name': 'arabic-reshaper',
            'description_ar': 'مكتبة لإعادة تشكيل النصوص العربية',
            'description_en': 'Library for reshaping Arabic text'
        },
        {
            'name': 'python-bidi',
            'description_ar': 'مكتبة لتطبيق خوارزمية اتجاه النص',
            'description_en': 'Library for applying text direction algorithm'
        }
    ]
    
    # تثبيت المكتبات
    success_count = 0
    for package in packages:
        if install_package(package['name'], package['description_ar'], package['description_en']):
            success_count += 1
    
    # فحص النتائج
    if success_count == len(packages):
        print(f"\n🎉 تم تثبيت جميع المكتبات بنجاح! ({success_count}/{len(packages)})")
        print(f"🎉 All libraries installed successfully! ({success_count}/{len(packages)})")
        
        # اختبار التثبيت
        if test_installation():
            # إنشاء ملف اختبار
            create_test_file()
            
            # عرض تعليمات الاستخدام
            show_usage_instructions()
            
            print("\n✅ تم إنجاز التثبيت بنجاح!")
            print("✅ Installation completed successfully!")
            print("🌟 نظام بصيرة الآن يدعم النصوص العربية بشكل صحيح!")
            print("🌟 Basira System now supports Arabic text correctly!")
            return True
        else:
            print("\n❌ فشل في اختبار التثبيت")
            print("❌ Installation test failed")
            return False
    else:
        print(f"\n⚠️ تم تثبيت {success_count} من {len(packages)} مكتبات فقط")
        print(f"⚠️ Only {success_count} out of {len(packages)} libraries installed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎯 يمكنك الآن تشغيل نظام بصيرة مع دعم كامل للنصوص العربية!")
            print("🎯 You can now run Basira System with full Arabic text support!")
        else:
            print("\n❌ حدث خطأ أثناء التثبيت")
            print("❌ An error occurred during installation")
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف التثبيت بواسطة المستخدم")
        print("🛑 Installation stopped by user")
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
