#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Text Handler for Basira System
معالج النصوص العربية لنظام بصيرة

This module handles Arabic text display issues in Python GUI applications,
specifically the right-to-left (RTL) text direction problem.

هذه الوحدة تعالج مشاكل عرض النصوص العربية في تطبيقات Python GUI،
خاصة مشكلة اتجاه النص من اليمين إلى اليسار.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import sys
import unicodedata
from typing import Optional

# محاولة استيراد مكتبات معالجة النصوص العربية
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_LIBS_AVAILABLE = True
    print("✅ مكتبات معالجة النصوص العربية متاحة")
except ImportError:
    ARABIC_LIBS_AVAILABLE = False
    print("⚠️ مكتبات معالجة النصوص العربية غير متاحة")
    print("💡 يمكن تثبيتها بـ: pip install arabic-reshaper python-bidi")


class ArabicTextHandler:
    """معالج النصوص العربية"""
    
    def __init__(self):
        """تهيئة معالج النصوص"""
        self.libs_available = ARABIC_LIBS_AVAILABLE
        
    def fix_arabic_text(self, text: str) -> str:
        """
        إصلاح اتجاه النص العربي
        Fix Arabic text direction
        """
        if not text:
            return text
            
        # إذا كانت المكتبات متاحة، استخدمها
        if self.libs_available:
            try:
                # إعادة تشكيل النص العربي
                reshaped_text = reshape(text)
                # تطبيق خوارزمية BiDi
                display_text = get_display(reshaped_text)
                return display_text
            except Exception as e:
                print(f"⚠️ خطأ في معالجة النص العربي: {e}")
                return self.fallback_fix(text)
        else:
            # استخدام الحل البديل
            return self.fallback_fix(text)
    
    def fallback_fix(self, text: str) -> str:
        """
        حل بديل لإصلاح النصوص العربية بدون مكتبات خارجية
        Fallback solution for Arabic text without external libraries
        """
        # فصل النص إلى أجزاء عربية وإنجليزية
        parts = []
        current_part = ""
        current_is_arabic = False
        
        for char in text:
            char_is_arabic = self.is_arabic_char(char)
            
            if char_is_arabic != current_is_arabic:
                if current_part:
                    if current_is_arabic:
                        # عكس ترتيب الكلمات العربية
                        words = current_part.split()
                        parts.append(' '.join(reversed(words)))
                    else:
                        parts.append(current_part)
                current_part = char
                current_is_arabic = char_is_arabic
            else:
                current_part += char
        
        # إضافة الجزء الأخير
        if current_part:
            if current_is_arabic:
                words = current_part.split()
                parts.append(' '.join(reversed(words)))
            else:
                parts.append(current_part)
        
        return ''.join(parts)
    
    def is_arabic_char(self, char: str) -> bool:
        """
        تحديد ما إذا كان الحرف عربياً
        Determine if character is Arabic
        """
        # نطاقات الأحرف العربية في Unicode
        arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]
        
        char_code = ord(char)
        for start, end in arabic_ranges:
            if start <= char_code <= end:
                return True
        return False
    
    def prepare_text_for_gui(self, text: str, widget_type: str = "label") -> str:
        """
        تحضير النص للعرض في واجهة المستخدم
        Prepare text for GUI display
        """
        # إصلاح اتجاه النص
        fixed_text = self.fix_arabic_text(text)
        
        # تطبيق تحسينات خاصة حسب نوع العنصر
        if widget_type == "button":
            # للأزرار، قد نحتاج مساحات إضافية
            fixed_text = f" {fixed_text} "
        elif widget_type == "title":
            # للعناوين، قد نضيف رموز تزيينية
            if self.contains_arabic(text):
                fixed_text = f"🌟 {fixed_text} 🌟"
        
        return fixed_text
    
    def contains_arabic(self, text: str) -> bool:
        """
        تحديد ما إذا كان النص يحتوي على أحرف عربية
        Check if text contains Arabic characters
        """
        return any(self.is_arabic_char(char) for char in text)
    
    def split_mixed_text(self, text: str) -> list:
        """
        تقسيم النص المختلط (عربي وإنجليزي) إلى أجزاء
        Split mixed text (Arabic and English) into parts
        """
        parts = []
        current_part = ""
        current_is_arabic = None
        
        for char in text:
            char_is_arabic = self.is_arabic_char(char)
            
            if current_is_arabic is None:
                current_is_arabic = char_is_arabic
                current_part = char
            elif char_is_arabic != current_is_arabic:
                parts.append({
                    'text': current_part,
                    'is_arabic': current_is_arabic
                })
                current_part = char
                current_is_arabic = char_is_arabic
            else:
                current_part += char
        
        if current_part:
            parts.append({
                'text': current_part,
                'is_arabic': current_is_arabic
            })
        
        return parts
    
    def format_for_console(self, text: str) -> str:
        """
        تنسيق النص للعرض في وحدة التحكم
        Format text for console display
        """
        # في وحدة التحكم، عادة لا نحتاج معالجة خاصة
        return text
    
    def install_arabic_libs(self) -> bool:
        """
        محاولة تثبيت مكتبات معالجة النصوص العربية
        Attempt to install Arabic text processing libraries
        """
        try:
            import subprocess
            import sys
            
            print("🔄 محاولة تثبيت مكتبات معالجة النصوص العربية...")
            
            # تثبيت المكتبات
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "arabic-reshaper", "python-bidi"
            ])
            
            # إعادة تحميل المكتبات
            global reshape, get_display, ARABIC_LIBS_AVAILABLE
            from arabic_reshaper import reshape
            from bidi.algorithm import get_display
            ARABIC_LIBS_AVAILABLE = True
            self.libs_available = True
            
            print("✅ تم تثبيت مكتبات معالجة النصوص العربية بنجاح!")
            return True
            
        except Exception as e:
            print(f"❌ فشل في تثبيت مكتبات معالجة النصوص العربية: {e}")
            return False


# إنشاء مثيل عام للاستخدام
arabic_handler = ArabicTextHandler()


def fix_arabic_text(text: str) -> str:
    """
    دالة سريعة لإصلاح النص العربي
    Quick function to fix Arabic text
    """
    return arabic_handler.fix_arabic_text(text)


def prepare_gui_text(text: str, widget_type: str = "label") -> str:
    """
    دالة سريعة لتحضير النص للواجهة
    Quick function to prepare text for GUI
    """
    return arabic_handler.prepare_text_for_gui(text, widget_type)


# دوال مساعدة للاستخدام السريع
def fix_button_text(text: str) -> str:
    """إصلاح نص الأزرار"""
    return prepare_gui_text(text, "button")


def fix_title_text(text: str) -> str:
    """إصلاح نص العناوين"""
    return prepare_gui_text(text, "title")


def fix_label_text(text: str) -> str:
    """إصلاح نص التسميات"""
    return prepare_gui_text(text, "label")


def test_arabic_handler():
    """اختبار معالج النصوص العربية"""
    print("🧪 اختبار معالج النصوص العربية...")
    
    test_texts = [
        "نظام بصيرة",
        "النظام المبتكر للتفاضل والتكامل",
        "Basira System نظام بصيرة",
        "تكامل = V × A، تفاضل = D × A",
        "إبداع باسل يحيى عبدالله من العراق/الموصل"
    ]
    
    for text in test_texts:
        print(f"\nالنص الأصلي: {text}")
        fixed = fix_arabic_text(text)
        print(f"النص المُصحح: {fixed}")
        
        # اختبار أنواع مختلفة من العناصر
        button_text = fix_button_text(text)
        title_text = fix_title_text(text)
        
        print(f"نص الزر: {button_text}")
        print(f"نص العنوان: {title_text}")
    
    print("\n✅ انتهى اختبار معالج النصوص العربية")


if __name__ == "__main__":
    # تشغيل الاختبار
    test_arabic_handler()
    
    # عرض معلومات المعالج
    print(f"\n📊 معلومات معالج النصوص العربية:")
    print(f"المكتبات المتقدمة متاحة: {'✅' if ARABIC_LIBS_AVAILABLE else '❌'}")
    
    if not ARABIC_LIBS_AVAILABLE:
        print("\n💡 لتحسين معالجة النصوص العربية، قم بتثبيت:")
        print("pip install arabic-reshaper python-bidi")
        
        # محاولة التثبيت التلقائي
        install_choice = input("\nهل تريد محاولة التثبيت التلقائي؟ (y/n): ")
        if install_choice.lower() in ['y', 'yes', 'نعم']:
            arabic_handler.install_arabic_libs()
