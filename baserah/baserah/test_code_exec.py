#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'baserah_system/code_execution')

try:
    from expert_guided_code_executor import ExpertGuidedCodeExecutor, CodeExecutionRequest, ProgrammingLanguage
    print("🎉 تم استيراد منفذ الأكواد الموجه بالخبير بنجاح!")
    
    # إنشاء المنفذ
    code_executor = ExpertGuidedCodeExecutor()
    print("✅ تم إنشاء منفذ الأكواد")
    
    # كود اختبار بسيط
    test_code = '''
def hello():
    print("Hello World")
    return "OK"

result = hello()
'''
    
    # طلب تنفيذ
    execution_request = CodeExecutionRequest(
        code=test_code,
        language=ProgrammingLanguage.PYTHON,
        test_cases=[{"input": "", "expected_output": "Hello World"}],
        quality_requirements={"performance": 0.8, "security": 0.9},
        auto_testing=True,
        security_check=True,
        performance_analysis=True,
        code_review=True
    )
    
    print("🚀 بدء تنفيذ الكود...")
    result = code_executor.execute_code_with_expert_guidance(execution_request)
    
    print(f"✅ النجاح: {result.success}")
    print(f"🎯 جودة الكود: {result.code_quality_score:.2%}")
    print(f"📋 موافق للتسليم: {'نعم' if result.approved_for_delivery else 'لا'}")
    
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()
