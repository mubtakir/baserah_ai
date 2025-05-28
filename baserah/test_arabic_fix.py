#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Arabic Text Fix for Basira System
اختبار إصلاح النصوص العربية لنظام بصيرة

Quick test to verify Arabic text handling is working correctly.
اختبار سريع للتأكد من عمل معالج النصوص العربية بشكل صحيح.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

try:
    from baserah_system.arabic_text_handler import fix_arabic_text, fix_button_text, fix_title_text, fix_label_text
    ARABIC_HANDLER_AVAILABLE = True
    print("✅ معالج النصوص العربية متاح")
except ImportError as e:
    print(f"❌ معالج النصوص العربية غير متاح: {e}")
    ARABIC_HANDLER_AVAILABLE = False
    # دوال بديلة
    def fix_arabic_text(text): return text
    def fix_button_text(text): return text
    def fix_title_text(text): return text
    def fix_label_text(text): return text


class ArabicTextTestApp:
    """تطبيق اختبار النصوص العربية"""

    def __init__(self):
        """تهيئة التطبيق"""
        self.root = tk.Tk()
        self.root.title("اختبار النصوص العربية - نظام بصيرة")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        self.create_widgets()

    def create_widgets(self):
        """إنشاء عناصر الواجهة"""
        
        # العنوان الرئيسي
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        # اختبار العنوان
        title_label = ttk.Label(title_frame,
                               text=fix_title_text("🌟 اختبار النصوص العربية - نظام بصيرة 🌟"),
                               font=('Arial', 16, 'bold'))
        title_label.pack()

        # معلومات الحالة
        status_frame = ttk.LabelFrame(self.root, text=fix_label_text("حالة معالج النصوص العربية"))
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        status_text = f"""
✅ معالج النصوص العربية: {'متاح' if ARABIC_HANDLER_AVAILABLE else 'غير متاح'}
🔧 Python tkinter: متاح
📅 وقت الاختبار: الآن
        """

        status_label = ttk.Label(status_frame, text=fix_arabic_text(status_text), font=('Arial', 11))
        status_label.pack(padx=10, pady=10)

        # اختبارات النصوص
        tests_frame = ttk.LabelFrame(self.root, text=fix_label_text("اختبارات النصوص العربية"))
        tests_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # النصوص التجريبية
        test_texts = [
            "نظام بصيرة",
            "النظام المبتكر للتفاضل والتكامل",
            "النظام الثوري لتفكيك الدوال",
            "إبداع باسل يحيى عبدالله من العراق/الموصل",
            "تكامل = V × A، تفاضل = D × A",
            "A = x.dA - ∫x.d2A"
        ]

        for i, text in enumerate(test_texts):
            # إطار لكل اختبار
            test_frame = ttk.Frame(tests_frame)
            test_frame.pack(fill=tk.X, padx=5, pady=2)

            # النص الأصلي
            original_label = ttk.Label(test_frame, text=f"الأصلي: {text}", 
                                     font=('Arial', 10), foreground='red')
            original_label.pack(anchor='w')

            # النص المُصحح
            fixed_text = fix_arabic_text(text)
            fixed_label = ttk.Label(test_frame, text=f"المُصحح: {fixed_text}", 
                                   font=('Arial', 10), foreground='green')
            fixed_label.pack(anchor='w')

            # خط فاصل
            if i < len(test_texts) - 1:
                separator = ttk.Separator(test_frame, orient='horizontal')
                separator.pack(fill=tk.X, pady=2)

        # أزرار الاختبار
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(pady=10)

        # اختبار الأزرار
        test_btn1 = ttk.Button(buttons_frame, text=fix_button_text("🚀 اختبار زر عربي"),
                              command=self.test_button_click)
        test_btn1.pack(side=tk.LEFT, padx=5)

        test_btn2 = ttk.Button(buttons_frame, text=fix_button_text("🧮 النظام المبتكر"),
                              command=self.test_button_click)
        test_btn2.pack(side=tk.LEFT, padx=5)

        test_btn3 = ttk.Button(buttons_frame, text=fix_button_text("🌟 التفكيك الثوري"),
                              command=self.test_button_click)
        test_btn3.pack(side=tk.LEFT, padx=5)

        # نتائج الاختبار
        result_frame = ttk.LabelFrame(self.root, text=fix_label_text("نتائج الاختبار"))
        result_frame.pack(fill=tk.X, padx=10, pady=5)

        self.result_text = tk.Text(result_frame, height=8, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, padx=5, pady=5)

        # إضافة نتائج أولية
        initial_result = fix_arabic_text("""
🎯 نتائج اختبار النصوص العربية:

✅ إذا كانت النصوص تظهر بالاتجاه الصحيح، فالمعالج يعمل بنجاح!
❌ إذا كانت النصوص معكوسة، فهناك مشكلة في المعالج.

🔧 حالة المعالج: """ + ('يعمل بنجاح' if ARABIC_HANDLER_AVAILABLE else 'يحتاج إصلاح'))

        self.result_text.insert(tk.END, initial_result)

        # شريط الحالة
        self.create_status_bar()

    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set(fix_arabic_text("جاهز للاختبار - انقر على الأزرار لاختبار النصوص العربية"))

        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)

        # معلومات إضافية
        info_label = ttk.Label(status_frame, text="Basira System - Arabic Text Test")
        info_label.pack(side=tk.RIGHT, padx=5)

    def test_button_click(self):
        """اختبار النقر على الأزرار"""
        import datetime
        
        test_message = fix_arabic_text(f"""
🎉 تم النقر على الزر بنجاح!
📅 الوقت: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ النصوص العربية تعمل بشكل صحيح!
🌟 نظام بصيرة - إبداع باسل يحيى عبدالله

""")
        
        self.result_text.insert(tk.END, test_message)
        self.result_text.see(tk.END)
        
        self.status_var.set(fix_arabic_text("✅ تم اختبار الزر بنجاح - النصوص العربية تعمل!"))

    def run(self):
        """تشغيل التطبيق"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    print("🧪 بدء اختبار النصوص العربية لنظام بصيرة...")
    print("🧪 Starting Arabic text test for Basira System...")
    
    # اختبار المعالج في وحدة التحكم أولاً
    print("\n📋 اختبار المعالج في وحدة التحكم:")
    test_texts = [
        "نظام بصيرة",
        "النظام المبتكر للتفاضل والتكامل",
        "إبداع باسل يحيى عبدالله"
    ]
    
    for text in test_texts:
        print(f"الأصلي: {text}")
        fixed = fix_arabic_text(text)
        print(f"المُصحح: {fixed}")
        print("---")
    
    print("\n🖥️ بدء اختبار الواجهة الرسومية...")
    
    try:
        app = ArabicTextTestApp()
        app.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل اختبار النصوص العربية: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
