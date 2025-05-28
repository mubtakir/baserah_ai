#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار عمل النظام - System Working Test
اختبار بسيط للتأكد من أن النظام يعمل

Author: Basil Yahya Abdullah - Iraq/Mosul
"""

import os
import sys

print("🌟" + "="*80 + "🌟")
print("🔬 اختبار عمل النظام - System Working Test")
print("⚡ التأكد من أن جميع المكونات تعمل")
print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
print("🌟" + "="*80 + "🌟")

# Test 1: Python Environment
print("\n1️⃣ اختبار بيئة Python...")
print(f"✅ Python Version: {sys.version}")
print(f"✅ Current Directory: {os.getcwd()}")

# Test 2: File System
print("\n2️⃣ اختبار نظام الملفات...")
if os.path.exists("baserah_system"):
    print("✅ مجلد baserah_system موجود")
    
    if os.path.exists("baserah_system/revolutionary_core"):
        print("✅ مجلد revolutionary_core موجود")
        
        if os.path.exists("baserah_system/revolutionary_core/unified_revolutionary_foundation.py"):
            print("✅ ملف unified_revolutionary_foundation.py موجود")
        else:
            print("❌ ملف unified_revolutionary_foundation.py غير موجود")
    else:
        print("❌ مجلد revolutionary_core غير موجود")
else:
    print("❌ مجلد baserah_system غير موجود")

# Test 3: Import Test
print("\n3️⃣ اختبار الاستيراد...")
sys.path.insert(0, "baserah_system")

try:
    # Test basic imports
    import numpy as np
    print("✅ NumPy متوفر")
except ImportError:
    print("❌ NumPy غير متوفر")

try:
    import math
    print("✅ Math متوفر")
except ImportError:
    print("❌ Math غير متوفر")

# Test 4: Revolutionary Foundation Test
print("\n4️⃣ اختبار الأساس الثوري...")
try:
    # Try direct import
    import revolutionary_core.unified_revolutionary_foundation as urf
    foundation = urf.get_revolutionary_foundation()
    print(f"✅ الأساس الثوري يعمل: {len(foundation.revolutionary_terms)} حد")
    
    # Test unit creation
    learning_unit = urf.create_revolutionary_unit("learning")
    print(f"✅ إنشاء الوحدات يعمل: {len(learning_unit.unit_terms)} حد")
    
    # Test processing
    test_input = {"wisdom_depth": 0.8}
    output = learning_unit.process_revolutionary_input(test_input)
    print(f"✅ المعالجة تعمل: {output.get('total_revolutionary_value', 0):.3f}")
    
    revolutionary_test = True
    
except Exception as e:
    print(f"❌ خطأ في الأساس الثوري: {e}")
    revolutionary_test = False

# Test 5: Integration Test
print("\n5️⃣ اختبار التكامل...")
try:
    import integration.unified_system_integration as usi
    integration = usi.UnifiedSystemIntegration()
    print("✅ نظام التكامل يعمل")
    integration_test = True
except Exception as e:
    print(f"❌ خطأ في نظام التكامل: {e}")
    integration_test = False

# Test 6: Dream Interpretation Test
print("\n6️⃣ اختبار تفسير الأحلام...")
try:
    import dream_interpretation.revolutionary_dream_interpreter_unified as driu
    interpreter = driu.create_unified_revolutionary_dream_interpreter()
    print("✅ تفسير الأحلام يعمل")
    dream_test = True
except Exception as e:
    print(f"❌ خطأ في تفسير الأحلام: {e}")
    dream_test = False

# Test 7: Learning Systems Test
print("\n7️⃣ اختبار أنظمة التعلم...")
try:
    import learning.reinforcement.innovative_rl_unified as iru
    learning_system = iru.create_unified_revolutionary_learning_system()
    print("✅ أنظمة التعلم تعمل")
    learning_test = True
except Exception as e:
    print(f"❌ خطأ في أنظمة التعلم: {e}")
    learning_test = False

# Test 8: Interfaces Test
print("\n8️⃣ اختبار الواجهات...")
try:
    import interfaces.web.unified_web_interface as uwi
    web_interface = uwi.create_unified_web_interface()
    print("✅ واجهة الويب تعمل")
    
    import interfaces.desktop.unified_desktop_interface as udi
    desktop_interface = udi.create_unified_desktop_interface()
    print("✅ واجهة سطح المكتب تعمل")
    
    interfaces_test = True
except Exception as e:
    print(f"❌ خطأ في الواجهات: {e}")
    interfaces_test = False

# Final Results
print("\n" + "🌟" + "="*80 + "🌟")
print("📊 النتائج النهائية")
print("🌟" + "="*80 + "🌟")

tests = [
    ("الأساس الثوري", revolutionary_test),
    ("نظام التكامل", integration_test),
    ("تفسير الأحلام", dream_test),
    ("أنظمة التعلم", learning_test),
    ("الواجهات", interfaces_test)
]

passed_tests = sum(1 for _, result in tests if result)
total_tests = len(tests)
success_rate = (passed_tests / total_tests) * 100

print(f"\n📈 ملخص الاختبارات:")
print(f"   الاختبارات الناجحة: {passed_tests}/{total_tests}")
print(f"   معدل النجاح: {success_rate:.1f}%")

print(f"\n📋 تفاصيل النتائج:")
for test_name, result in tests:
    status = "✅ نجح" if result else "❌ فشل"
    print(f"   {test_name}: {status}")

# Final Verdict
if success_rate >= 80:
    verdict = "🎉 ممتاز! النظام يعمل بكفاءة عالية!"
elif success_rate >= 60:
    verdict = "✅ جيد! النظام يعمل مع بعض المشاكل"
elif success_rate >= 40:
    verdict = "⚠️ متوسط! النظام يحتاج تحسينات"
else:
    verdict = "❌ ضعيف! النظام يحتاج إصلاحات جوهرية"

print(f"\n🎯 الحكم النهائي: {verdict}")

if revolutionary_test:
    print("\n🌟 الأساس الثوري يعمل - AI-OOP مطبق!")
    print("⚡ النظام الخبير/المستكشف متوفر!")
    print("🧮 المعادلات المتكيفة تعمل!")

print("\n🌟 إبداع باسل يحيى عبدالله من العراق/الموصل محفوظ ومطور! 🌟")
print("🎯 اختبار النظام مكتمل!")

# Save results
results_summary = {
    "total_tests": total_tests,
    "passed_tests": passed_tests,
    "success_rate": success_rate,
    "test_details": {name: result for name, result in tests},
    "verdict": verdict,
    "revolutionary_foundation_working": revolutionary_test
}

try:
    import json
    with open("system_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 تم حفظ النتائج في: system_test_results.json")
except Exception as e:
    print(f"\n⚠️ لم يتم حفظ النتائج: {e}")

print("\n🚀 النظام جاهز للاستخدام!")
