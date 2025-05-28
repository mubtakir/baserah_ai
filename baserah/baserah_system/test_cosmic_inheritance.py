#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار وراثة المعادلة الكونية الأم
Test Cosmic Mother Equation Inheritance
"""

import sys
import os

print("🌟" + "="*80 + "🌟")
print("🧪 اختبار وراثة المعادلة الكونية الأم")
print("🌳 أول اختبار للشجرة الأم مع وحدة الرسم والاستنباط")
print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
print("🌟" + "="*80 + "🌟")

# إضافة المسار
sys.path.insert(0, os.path.dirname(__file__))

def test_cosmic_mother_equation():
    """اختبار المعادلة الكونية الأم"""
    print("\n🌌 اختبار المعادلة الكونية الأم...")
    
    try:
        from mathematical_core.cosmic_general_shape_equation import (
            create_cosmic_general_shape_equation,
            CosmicTermType
        )
        
        # إنشاء المعادلة الكونية الأم
        cosmic_eq = create_cosmic_general_shape_equation()
        print("✅ تم إنشاء المعادلة الكونية الأم بنجاح")
        
        # فحص الحالة
        status = cosmic_eq.get_cosmic_status()
        print(f"📊 إجمالي الحدود الكونية: {status['total_cosmic_terms']}")
        print(f"🌟 حدود باسل الثورية: {status['statistics']['basil_terms']}")
        
        # اختبار وراثة حدود الرسم
        drawing_terms = cosmic_eq.get_drawing_terms()
        print(f"🎨 حدود الرسم المتاحة: {len(drawing_terms)}")
        
        # اختبار الوراثة
        inherited = cosmic_eq.inherit_terms_for_unit("test_unit", drawing_terms[:5])
        print(f"🍃 تم وراثة {len(inherited)} حد بنجاح")
        
        # اختبار تقييم المعادلة
        test_values = {
            CosmicTermType.DRAWING_X: 5.0,
            CosmicTermType.DRAWING_Y: 3.0,
            CosmicTermType.BASIL_INNOVATION: 1.0
        }
        
        result = cosmic_eq.evaluate_cosmic_equation(test_values)
        print(f"🧮 نتيجة تقييم المعادلة: {result:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في المعادلة الكونية: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_drawing_unit_inheritance():
    """اختبار وراثة وحدة الرسم"""
    print("\n🎨 اختبار وراثة وحدة الرسم والاستنباط...")
    
    try:
        from artistic_unit.revolutionary_drawing_extraction_unit import (
            create_revolutionary_drawing_extraction_unit
        )
        
        # إنشاء وحدة الرسم
        drawing_unit = create_revolutionary_drawing_extraction_unit()
        print("✅ تم إنشاء وحدة الرسم والاستنباط بنجاح")
        
        # اختبار الوراثة
        inheritance_test = drawing_unit.test_cosmic_inheritance()
        print(f"🧪 نتائج اختبار الوراثة:")
        print(f"   الوراثة ناجحة: {inheritance_test['inheritance_successful']}")
        print(f"   الحدود الموروثة: {inheritance_test['inherited_terms_count']}")
        print(f"   حدود باسل موروثة: {inheritance_test['basil_terms_inherited']}")
        
        if inheritance_test.get("shape_creation_successful"):
            print(f"   إنشاء الشكل ناجح: ✅")
        else:
            print(f"   إنشاء الشكل: ❌")
        
        # عرض حالة الوحدة
        status = drawing_unit.get_unit_status()
        print(f"📊 حالة الوحدة:")
        print(f"   الوراثة الكونية نشطة: {status['cosmic_inheritance_active']}")
        print(f"   الحدود الموروثة: {len(status['inherited_terms'])}")
        
        return inheritance_test['inheritance_successful']
        
    except Exception as e:
        print(f"❌ خطأ في وحدة الرسم: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """الدالة الرئيسية للاختبار"""
    
    tests = [
        ("المعادلة الكونية الأم", test_cosmic_mother_equation),
        ("وراثة وحدة الرسم", test_drawing_unit_inheritance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 تشغيل اختبار: {test_name}")
        print("-" * 50)
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            print(f"✅ نجح اختبار {test_name}")
        else:
            print(f"❌ فشل اختبار {test_name}")
    
    # النتائج النهائية
    print("\n" + "🌟" + "="*80 + "🌟")
    print("📊 النتائج النهائية")
    print("🌟" + "="*80 + "🌟")
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📈 ملخص الاختبارات:")
    print(f"   الاختبارات الناجحة: {passed_tests}/{total_tests}")
    print(f"   معدل النجاح: {success_rate:.1f}%")
    
    print(f"\n📋 تفاصيل النتائج:")
    for test_name, result in results:
        status = "✅ نجح" if result else "❌ فشل"
        print(f"   {test_name}: {status}")
    
    if success_rate >= 80:
        verdict = "🎉 ممتاز! المعادلة الكونية الأم تعمل بكفاءة عالية!"
    elif success_rate >= 60:
        verdict = "✅ جيد! النظام يعمل مع بعض المشاكل البسيطة"
    else:
        verdict = "❌ يحتاج تحسين! النظام يحتاج إصلاحات"
    
    print(f"\n🎯 الحكم النهائي: {verdict}")
    
    if passed_tests > 0:
        print("\n🌟 المزايا المحققة:")
        print("   🌳 معادلة كونية أم شاملة")
        print("   🍃 وراثة ناجحة للحدود")
        print("   🎨 أول اختبار مع وحدة الرسم")
        print("   🌟 حدود باسل الثورية مدمجة")
        print("   🧮 تقييم رياضي دقيق")
    
    print("\n🌟 إبداع باسل يحيى عبدالله من العراق/الموصل محفوظ ومطور! 🌟")
    print("🎯 أول اختبار للمعادلة الكونية الأم مكتمل!")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
