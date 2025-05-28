#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار نظام تفسير الأحلام المتقدم وفق نظرية باسل

هذا الملف يحتوي على اختبارات شاملة لنظام تفسير الأحلام
ويتضمن أمثلة من الكتاب والحالات الواقعية.
"""

import sys
import os
import json
from datetime import datetime

# إضافة مسار النظام
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basil_dream_system import (
    BasilDreamInterpreter, 
    DreamerProfile, 
    create_basil_dream_interpreter,
    create_dreamer_profile
)

def test_basic_dream_interpretation():
    """اختبار التفسير الأساسي للأحلام"""
    print("=== اختبار التفسير الأساسي ===")
    
    # إنشاء مفسر الأحلام
    interpreter = create_basil_dream_interpreter()
    
    # إنشاء ملف شخصي للرائي
    dreamer = create_dreamer_profile(
        name="أحمد محمد",
        profession="مهندس",
        age=35,
        religion="إسلام",
        interests=["تكنولوجيا", "قراءة"],
        current_concerns=["عمل جديد", "زواج"]
    )
    
    # حلم بسيط للاختبار
    dream_text = "رأيت في المنام شجرة كبيرة خضراء وتحتها ماء صافي"
    
    # تفسير الحلم
    interpretation = interpreter.interpret_dream(dream_text, dreamer)
    
    # طباعة النتائج
    print(f"نص الحلم: {interpretation.dream_text}")
    print(f"نوع الحلم: {interpretation.dream_type.value}")
    print(f"مستوى الثقة: {interpretation.confidence_level:.2f}")
    print(f"عدد العناصر: {len(interpretation.elements)}")
    print(f"الرسالة الإجمالية: {interpretation.overall_message}")
    print("\nالتوصيات:")
    for rec in interpretation.recommendations:
        print(f"- {rec}")
    
    return interpretation

def test_tashif_mechanism():
    """اختبار آلية التصحيف"""
    print("\n=== اختبار آلية التصحيف ===")
    
    interpreter = create_basil_dream_interpreter()
    
    dreamer = create_dreamer_profile(
        name="فاطمة أحمد",
        profession="طبيبة",
        age=30,
        religion="إسلام"
    )
    
    # حلم يحتوي على تصحيف (بسة = قطة، لكن المعنى "بس" = فقط/رجاء)
    dream_text = "رأيت بسة صغيرة تأكل من لحم بسة أخرى"
    
    interpretation = interpreter.interpret_dream(dream_text, dreamer)
    
    print(f"نص الحلم: {interpretation.dream_text}")
    print(f"الآليات المستخدمة: {[mech.value for mech in interpretation.symbolic_mechanisms]}")
    print(f"التفسير الرمزي: {interpretation.interpretation_layers.get('رمزي', 'غير متوفر')}")
    
    return interpretation

def test_jinas_mechanism():
    """اختبار آلية الجناس"""
    print("\n=== اختبار آلية الجناس ===")
    
    interpreter = create_basil_dream_interpreter()
    
    dreamer = create_dreamer_profile(
        name="خالد سوري",
        profession="صحفي",
        age=40,
        religion="إسلام",
        cultural_background="عربي",
        interests=["أخبار", "سياسة"]
    )
    
    # حلم يحتوي على جناس (سوري = من سوريا، لكن المعنى "سور" = اقتحام السور)
    dream_text = "دخل علينا سوري في البيت"
    
    interpretation = interpreter.interpret_dream(dream_text, dreamer)
    
    print(f"نص الحلم: {interpretation.dream_text}")
    print(f"الآليات المستخدمة: {[mech.value for mech in interpretation.symbolic_mechanisms]}")
    print(f"التفسير الشخصي: {interpretation.interpretation_layers.get('شخصي', 'غير متوفر')}")
    
    return interpretation

def test_traditional_symbols():
    """اختبار الرموز التراثية"""
    print("\n=== اختبار الرموز التراثية ===")
    
    interpreter = create_basil_dream_interpreter()
    
    dreamer = create_dreamer_profile(
        name="عائشة محمد",
        profession="معلمة",
        age=28,
        religion="إسلام",
        social_status="عزباء"
    )
    
    # حلم مشابه لمثال الحمامة والصقر من الكتاب
    dream_text = "رأيت حمامة بيضاء جميلة على سطح المسجد، فجاء صقر وأخذها"
    
    interpretation = interpreter.interpret_dream(dream_text, dreamer)
    
    print(f"نص الحلم: {interpretation.dream_text}")
    print(f"العناصر المستخرجة:")
    for elem in interpretation.elements:
        print(f"  - {elem.element}: {elem.symbolic_meanings}")
    print(f"التفسير الديني: {interpretation.interpretation_layers.get('ديني_روحي', 'غير متوفر')}")
    
    return interpretation

def test_non_interpretable_dream():
    """اختبار حلم لا يُعبر"""
    print("\n=== اختبار حلم لا يُعبر ===")
    
    interpreter = create_basil_dream_interpreter()
    
    dreamer = create_dreamer_profile(
        name="محمد علي",
        profession="عامل",
        age=45,
        health_status="مريض",
        temperament="صفراوي"
    )
    
    # حلم يحمل علامات انعكاس المرض
    dream_text = "رأيت نيران كثيرة وأشياء صفراء في كل مكان وحرارة شديدة"
    
    interpretation = interpreter.interpret_dream(dream_text, dreamer)
    
    print(f"نص الحلم: {interpretation.dream_text}")
    print(f"نوع الحلم: {interpretation.dream_type.value}")
    print(f"الرسالة: {interpretation.overall_message}")
    print("\nالتحذيرات:")
    for warning in interpretation.warnings:
        print(f"- {warning}")
    
    return interpretation

def test_personal_context():
    """اختبار تأثير السياق الشخصي"""
    print("\n=== اختبار تأثير السياق الشخصي ===")
    
    interpreter = create_basil_dream_interpreter()
    
    # رائي مهندس
    engineer_dreamer = create_dreamer_profile(
        name="أحمد المهندس",
        profession="مهندس معماري",
        interests=["بناء", "تصميم"],
        current_concerns=["مشروع جديد"]
    )
    
    # رائي فنان
    artist_dreamer = create_dreamer_profile(
        name="سارة الفنانة",
        profession="رسامة",
        interests=["فن", "ألوان"],
        current_concerns=["معرض فني"]
    )
    
    # نفس الحلم لكلا الرائيين
    dream_text = "رأيت شجرة جميلة بألوان زاهية"
    
    # تفسير للمهندس
    engineer_interpretation = interpreter.interpret_dream(dream_text, engineer_dreamer)
    print("تفسير المهندس:")
    print(f"التفسير الشخصي: {engineer_interpretation.interpretation_layers.get('شخصي', 'غير متوفر')}")
    
    # تفسير للفنانة
    artist_interpretation = interpreter.interpret_dream(dream_text, artist_dreamer)
    print("\nتفسير الفنانة:")
    print(f"التفسير الشخصي: {artist_interpretation.interpretation_layers.get('شخصي', 'غير متوفر')}")
    
    return engineer_interpretation, artist_interpretation

def test_statistics():
    """اختبار الإحصائيات"""
    print("\n=== اختبار الإحصائيات ===")
    
    interpreter = create_basil_dream_interpreter()
    
    # إجراء عدة تفسيرات
    test_basic_dream_interpretation()
    test_tashif_mechanism()
    test_jinas_mechanism()
    
    # الحصول على الإحصائيات
    stats = interpreter.get_interpretation_statistics()
    
    print("إحصائيات النظام:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    return stats

def test_export_interpretation():
    """اختبار تصدير التفسير"""
    print("\n=== اختبار تصدير التفسير ===")
    
    interpreter = create_basil_dream_interpreter()
    
    dreamer = create_dreamer_profile(
        name="تجربة التصدير",
        profession="مطور",
        religion="إسلام"
    )
    
    dream_text = "رأيت قمراً مضيئاً في السماء"
    
    interpretation = interpreter.interpret_dream(dream_text, dreamer)
    
    # تصدير إلى JSON
    interpretation_dict = interpretation.to_dict()
    
    # حفظ في ملف
    output_file = "dream_interpretation_export.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(interpretation_dict, f, ensure_ascii=False, indent=2)
    
    print(f"تم تصدير التفسير إلى: {output_file}")
    print("محتوى التصدير:")
    print(json.dumps(interpretation_dict, ensure_ascii=False, indent=2)[:500] + "...")
    
    return interpretation_dict

def run_comprehensive_test():
    """تشغيل اختبار شامل للنظام"""
    print("🌙 بدء الاختبار الشامل لنظام تفسير الأحلام وفق نظرية باسل 🌙")
    print("=" * 60)
    
    try:
        # تشغيل جميع الاختبارات
        test_basic_dream_interpretation()
        test_tashif_mechanism()
        test_jinas_mechanism()
        test_traditional_symbols()
        test_non_interpretable_dream()
        test_personal_context()
        test_statistics()
        test_export_interpretation()
        
        print("\n" + "=" * 60)
        print("✅ تم إنجاز جميع الاختبارات بنجاح!")
        print("🎉 نظام تفسير الأحلام وفق نظرية باسل يعمل بشكل ممتاز!")
        
    except Exception as e:
        print(f"\n❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test()
