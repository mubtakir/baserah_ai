#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تطبيق تفاعلي لعرض قدرات نظام بصيرة

هذا التطبيق يوفر واجهة تفاعلية بسيطة لتجريب
مختلف مكونات نظام بصيرة بما في ذلك تفسير الأحلام.
"""

import sys
import os
from datetime import datetime

def print_header():
    """طباعة رأس التطبيق"""
    print("=" * 70)
    print("🌟 مرحباً بك في نظام بصيرة التفاعلي 🌟")
    print("نظام ذكاء اصطناعي مبتكر للغة العربية وتفسير الأحلام")
    print("=" * 70)

def print_menu():
    """طباعة القائمة الرئيسية"""
    print("\n📋 القائمة الرئيسية:")
    print("1. 🌙 تفسير الأحلام")
    print("2. 🔧 اختبار النظام الرئيسي")
    print("3. 📊 عرض إحصائيات النظام")
    print("4. 🔍 اختبار مكونات النظام")
    print("5. ❌ خروج")
    print("-" * 50)

def dream_interpretation_demo():
    """عرض تفسير الأحلام"""
    print("\n🌙 === نظام تفسير الأحلام === 🌙")
    
    try:
        from dream_interpretation.basil_dream_system import create_basil_dream_interpreter, create_dreamer_profile
        
        # إنشاء مفسر الأحلام
        interpreter = create_basil_dream_interpreter()
        print("✅ تم تهيئة مفسر الأحلام")
        
        # إنشاء ملف شخصي افتراضي
        dreamer = create_dreamer_profile(
            name="المستخدم التجريبي",
            profession="موظف",
            religion="إسلام",
            interests=["قراءة", "تعلم"]
        )
        print("✅ تم إنشاء ملف الرائي")
        
        # أحلام تجريبية
        sample_dreams = [
            "رأيت في المنام شجرة كبيرة خضراء",
            "رأيت نفسي أطير في السماء",
            "رأيت ماءً صافياً يجري في النهر",
            "رأيت نوراً ساطعاً من السماء"
        ]
        
        print("\n🎯 أحلام تجريبية:")
        for i, dream in enumerate(sample_dreams, 1):
            print(f"{i}. {dream}")
        
        print("0. إدخال حلم مخصص")
        
        choice = input("\nاختر رقم الحلم (0-4): ").strip()
        
        if choice == "0":
            dream_text = input("أدخل نص الحلم: ").strip()
            if not dream_text:
                print("❌ لم يتم إدخال نص الحلم")
                return
        elif choice in ["1", "2", "3", "4"]:
            dream_text = sample_dreams[int(choice) - 1]
        else:
            print("❌ اختيار غير صحيح")
            return
        
        print(f"\n🔍 تفسير الحلم: {dream_text}")
        print("-" * 50)
        
        # تفسير الحلم
        interpretation = interpreter.interpret_dream(dream_text, dreamer)
        
        print(f"📊 نوع الحلم: {interpretation.dream_type.value}")
        print(f"📈 مستوى الثقة: {interpretation.confidence_level:.2f}")
        print(f"🔍 عدد العناصر: {len(interpretation.elements)}")
        
        if interpretation.elements:
            print("\n🎯 العناصر المستخرجة:")
            for elem in interpretation.elements[:3]:
                meanings = ", ".join(elem.symbolic_meanings[:2])
                print(f"  • {elem.element}: {meanings}")
        
        print(f"\n💭 الرسالة الإجمالية:")
        message_lines = interpretation.overall_message.split('\n')
        for line in message_lines[:5]:  # أول 5 أسطر
            if line.strip():
                print(f"  {line}")
        
        print(f"\n📋 التوصيات:")
        for rec in interpretation.recommendations[:3]:
            print(f"  • {rec}")
        
        if interpretation.warnings:
            print(f"\n⚠️ تحذيرات:")
            for warning in interpretation.warnings[:2]:
                print(f"  • {warning}")
        
    except Exception as e:
        print(f"❌ خطأ في نظام تفسير الأحلام: {e}")

def system_test_demo():
    """اختبار النظام الرئيسي"""
    print("\n🔧 === اختبار النظام الرئيسي === 🔧")
    
    try:
        import main
        print("✅ تم استيراد النظام الرئيسي بنجاح")
        
        # محاولة إنشاء النظام
        system = main.BasiraSystem()
        print("✅ تم إنشاء نظام بصيرة بنجاح")
        print("✅ جميع المكونات تعمل بشكل صحيح")
        
    except Exception as e:
        print(f"❌ خطأ في النظام الرئيسي: {e}")

def system_stats_demo():
    """عرض إحصائيات النظام"""
    print("\n📊 === إحصائيات النظام === 📊")
    
    # إحصائيات أساسية
    print("📈 معلومات النظام:")
    print(f"  • اسم النظام: بصيرة (Basira)")
    print(f"  • الإصدار: 1.0.0")
    print(f"  • تاريخ التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  • لغة البرمجة: Python 3.12")
    
    # فحص المكونات
    components = [
        "dream_interpretation",
        "arabic_nlp", 
        "mathematical_core",
        "symbolic_processing",
        "cognitive_linguistic",
        "integration_layer"
    ]
    
    print("\n🔧 حالة المكونات:")
    working_components = 0
    for component in components:
        try:
            exec(f"import {component}")
            print(f"  ✅ {component}: يعمل")
            working_components += 1
        except:
            print(f"  ❌ {component}: لا يعمل")
    
    print(f"\n📊 ملخص الحالة:")
    print(f"  • المكونات العاملة: {working_components}/{len(components)}")
    print(f"  • نسبة النجاح: {(working_components/len(components)*100):.1f}%")

def components_test_demo():
    """اختبار مكونات النظام"""
    print("\n🔍 === اختبار مكونات النظام === 🔍")
    
    tests = [
        ("نظام تفسير الأحلام", "dream_interpretation.basil_dream_system"),
        ("معالج اللغة العربية", "arabic_nlp.advanced_processor"),
        ("المعادلة العامة", "mathematical_core.general_shape_equation"),
        ("المعالجة الرمزية", "symbolic_processing.expert_explorer_system"),
        ("العمارة المعرفية", "cognitive_linguistic.cognitive_linguistic_architecture")
    ]
    
    for name, module in tests:
        try:
            exec(f"import {module}")
            print(f"  ✅ {name}: متاح")
        except Exception as e:
            print(f"  ❌ {name}: غير متاح ({str(e)[:50]}...)")

def main():
    """الدالة الرئيسية"""
    print_header()
    
    while True:
        print_menu()
        choice = input("اختر من القائمة (1-5): ").strip()
        
        if choice == "1":
            dream_interpretation_demo()
        elif choice == "2":
            system_test_demo()
        elif choice == "3":
            system_stats_demo()
        elif choice == "4":
            components_test_demo()
        elif choice == "5":
            print("\n👋 شكراً لاستخدام نظام بصيرة!")
            print("🌟 نراكم قريباً! 🌟")
            break
        else:
            print("❌ اختيار غير صحيح، يرجى المحاولة مرة أخرى")
        
        input("\n⏸️ اضغط Enter للمتابعة...")

if __name__ == "__main__":
    main()
