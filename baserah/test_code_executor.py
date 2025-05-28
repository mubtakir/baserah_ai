#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار منفذ الأكواد الموجه بالخبير
Test Expert-Guided Code Executor
"""

import sys
import os

# إضافة المسار
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baserah_system', 'code_execution'))

try:
    from expert_guided_code_executor import (
        ExpertGuidedCodeExecutor, 
        CodeExecutionRequest, 
        ProgrammingLanguage
    )
    
    print("🎉 تم استيراد منفذ الأكواد الموجه بالخبير بنجاح!")
    
    # إنشاء المنفذ
    code_executor = ExpertGuidedCodeExecutor()
    
    # كود اختبار بسيط
    test_code = '''
def hello_world():
    """دالة بسيطة للاختبار"""
    print("Hello, World!")
    return "success"

# تنفيذ الدالة
result = hello_world()
print(f"Result: {result}")
'''
    
    # طلب تنفيذ
    execution_request = CodeExecutionRequest(
        code=test_code,
        language=ProgrammingLanguage.PYTHON,
        test_cases=[
            {"input": "", "expected_output": "Hello, World!\\nResult: success"}
        ],
        quality_requirements={"performance": 0.8, "security": 0.9, "maintainability": 0.7},
        auto_testing=True,
        security_check=True,
        performance_analysis=True,
        code_review=True
    )
    
    print("✅ تم إنشاء طلب التنفيذ")
    print("🚀 بدء اختبار منفذ الأكواد...")
    
    # تنفيذ الكود
    result = code_executor.execute_code_with_expert_guidance(execution_request)
    
    print(f"\n💻 نتائج تنفيذ الكود:")
    print(f"   ✅ النجاح: {result.success}")
    print(f"   🎯 جودة الكود: {result.code_quality_score:.2%}")
    print(f"   🔒 نتيجة الأمان: {result.security_analysis['security_score']:.2%}")
    print(f"   ⚡ الأداء الإجمالي: {result.performance_metrics['overall_performance']:.2%}")
    print(f"   🧪 الاختبارات: {len([t for t in result.test_results if t['result'] == 'passed'])}/{len(result.test_results)} نجح")
    print(f"   📋 موافق للتسليم: {'✅ نعم' if result.approved_for_delivery else '❌ لا'}")
    
    if result.recommendations:
        print(f"\n💡 التوصيات:")
        for rec in result.recommendations:
            print(f"   • {rec}")
    
    print(f"\n📊 إحصائيات المنفذ:")
    print(f"   💻 معادلات برمجية: {len(code_executor.code_equations)}")
    print(f"   📚 قاعدة التعلم: {len(code_executor.code_learning_database)} لغة")
    
    print("\n🎉 اختبار منفذ الأكواد الموجه بالخبير مكتمل بنجاح!")

except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    print("🔍 التحقق من وجود الملفات...")
    
    # فحص وجود الملفات
    code_executor_path = os.path.join(os.path.dirname(__file__), 'baserah_system', 'code_execution', 'expert_guided_code_executor.py')
    if os.path.exists(code_executor_path):
        print("✅ ملف منفذ الأكواد موجود")
    else:
        print("❌ ملف منفذ الأكواد غير موجود")

except Exception as e:
    print(f"❌ خطأ عام: {e}")
    import traceback
    traceback.print_exc()
