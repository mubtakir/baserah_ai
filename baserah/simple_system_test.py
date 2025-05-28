#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار بسيط للنظام الموحد - Simple System Test
اختبار أساسي لمكونات نظام بصيرة الموحد

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - Simple Test
"""

import os
import sys
import time
from datetime import datetime

# Add baserah_system to path
sys.path.insert(0, 'baserah_system')

print("🌟" + "="*80 + "🌟")
print("🔬 اختبار بسيط للنظام الموحد - Simple System Test")
print("⚡ اختبار أساسي لمكونات نظام بصيرة")
print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
print("🌟" + "="*80 + "🌟")

def test_revolutionary_foundation():
    """اختبار الأساس الثوري"""
    print("\n🏗️ اختبار الأساس الثوري...")
    
    try:
        from revolutionary_core.unified_revolutionary_foundation import get_revolutionary_foundation
        foundation = get_revolutionary_foundation()
        print(f"✅ الأساس الثوري متوفر مع {len(foundation.revolutionary_terms)} حد ثوري")
        return True
    except Exception as e:
        print(f"❌ خطأ في الأساس الثوري: {e}")
        return False

def test_integration_system():
    """اختبار نظام التكامل"""
    print("\n🔗 اختبار نظام التكامل...")
    
    try:
        from integration.unified_system_integration import UnifiedSystemIntegration
        integration = UnifiedSystemIntegration()
        print("✅ نظام التكامل تم إنشاؤه بنجاح")
        return True
    except Exception as e:
        print(f"❌ خطأ في نظام التكامل: {e}")
        return False

def test_revolutionary_learning():
    """اختبار التعلم الثوري"""
    print("\n🧠 اختبار التعلم الثوري...")
    
    try:
        from learning.reinforcement.innovative_rl_unified import create_unified_revolutionary_learning_system
        learning_system = create_unified_revolutionary_learning_system()
        
        # اختبار قرار بسيط
        situation = {"complexity": 0.5, "novelty": 0.5}
        decision = learning_system.make_expert_decision(situation)
        
        print(f"✅ التعلم الثوري يعمل - قرار: {decision.get('decision', 'غير محدد')}")
        return True
    except Exception as e:
        print(f"❌ خطأ في التعلم الثوري: {e}")
        return False

def test_adaptive_equations():
    """اختبار المعادلات المتكيفة"""
    print("\n📐 اختبار المعادلات المتكيفة...")
    
    try:
        from learning.reinforcement.equation_based_rl_unified import create_unified_adaptive_equation_system
        equation_system = create_unified_adaptive_equation_system()
        
        # اختبار حل نمط بسيط
        pattern = [1, 2, 3]
        solution = equation_system.solve_pattern(pattern)
        
        print(f"✅ المعادلات المتكيفة تعمل - حل: {solution.get('solution', 'غير محدد')}")
        return True
    except Exception as e:
        print(f"❌ خطأ في المعادلات المتكيفة: {e}")
        return False

def test_revolutionary_agent():
    """اختبار الوكيل الثوري"""
    print("\n🤖 اختبار الوكيل الثوري...")
    
    try:
        from learning.innovative_reinforcement.agent_unified import create_unified_revolutionary_agent
        agent = create_unified_revolutionary_agent()
        
        # اختبار قرار بسيط
        situation = {
            "complexity": 0.6,
            "urgency": 0.4,
            "available_options": ["option_a", "option_b"]
        }
        decision = agent.make_revolutionary_decision(situation)
        
        print(f"✅ الوكيل الثوري يعمل - قرار: {decision.decision}")
        return True
    except Exception as e:
        print(f"❌ خطأ في الوكيل الثوري: {e}")
        return False

def test_dream_interpretation():
    """اختبار تفسير الأحلام"""
    print("\n🌙 اختبار تفسير الأحلام...")
    
    try:
        from dream_interpretation.revolutionary_dream_interpreter_unified import create_unified_revolutionary_dream_interpreter
        interpreter = create_unified_revolutionary_dream_interpreter()
        
        # اختبار تفسير بسيط
        dream = "رأيت ماء صافياً"
        profile = {"name": "باسل", "age": 30}
        
        interpretation = interpreter.interpret_dream_revolutionary(dream, profile)
        
        print(f"✅ تفسير الأحلام يعمل - ثقة: {interpretation.confidence_level:.2f}")
        return True
    except Exception as e:
        print(f"❌ خطأ في تفسير الأحلام: {e}")
        return False

def test_web_interface():
    """اختبار واجهة الويب"""
    print("\n🌐 اختبار واجهة الويب...")
    
    try:
        from interfaces.web.unified_web_interface import create_unified_web_interface
        web_interface = create_unified_web_interface(host='127.0.0.1', port=5001)
        print("✅ واجهة الويب تم إنشاؤها بنجاح")
        return True
    except Exception as e:
        print(f"❌ خطأ في واجهة الويب: {e}")
        return False

def test_desktop_interface():
    """اختبار واجهة سطح المكتب"""
    print("\n🖥️ اختبار واجهة سطح المكتب...")
    
    try:
        from interfaces.desktop.unified_desktop_interface import create_unified_desktop_interface
        desktop_interface = create_unified_desktop_interface()
        print("✅ واجهة سطح المكتب تم إنشاؤها بنجاح")
        return True
    except Exception as e:
        print(f"❌ خطأ في واجهة سطح المكتب: {e}")
        return False

def main():
    """الدالة الرئيسية"""
    start_time = time.time()
    
    # قائمة الاختبارات
    tests = [
        ("الأساس الثوري", test_revolutionary_foundation),
        ("نظام التكامل", test_integration_system),
        ("التعلم الثوري", test_revolutionary_learning),
        ("المعادلات المتكيفة", test_adaptive_equations),
        ("الوكيل الثوري", test_revolutionary_agent),
        ("تفسير الأحلام", test_dream_interpretation),
        ("واجهة الويب", test_web_interface),
        ("واجهة سطح المكتب", test_desktop_interface)
    ]
    
    # تشغيل الاختبارات
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ خطأ في اختبار {test_name}: {e}")
            results[test_name] = False
    
    # النتائج النهائية
    print("\n" + "🌟" + "="*80 + "🌟")
    print("📊 نتائج الاختبار البسيط")
    print("🌟" + "="*80 + "🌟")
    
    successful_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"\n📈 ملخص النتائج:")
    print(f"   إجمالي الاختبارات: {total_tests}")
    print(f"   الاختبارات الناجحة: {successful_tests}")
    print(f"   الاختبارات الفاشلة: {total_tests - successful_tests}")
    print(f"   معدل النجاح: {success_rate:.1f}%")
    print(f"   مدة الاختبار: {time.time() - start_time:.2f} ثانية")
    
    print(f"\n📋 تفاصيل النتائج:")
    for test_name, result in results.items():
        status = "✅ نجح" if result else "❌ فشل"
        print(f"   {test_name}: {status}")
    
    # الحكم النهائي
    if success_rate >= 80:
        verdict = "🎉 ممتاز! النظام يعمل بكفاءة عالية!"
    elif success_rate >= 60:
        verdict = "✅ جيد! النظام يعمل مع بعض المشاكل البسيطة"
    elif success_rate >= 40:
        verdict = "⚠️ متوسط! النظام يحتاج تحسينات"
    else:
        verdict = "❌ ضعيف! النظام يحتاج إصلاحات جوهرية"
    
    print(f"\n🎯 الحكم النهائي: {verdict}")
    
    print("\n🌟 إبداع باسل يحيى عبدالله من العراق/الموصل محفوظ ومطور! 🌟")
    print("🎯 اختبار النظام البسيط مكتمل!")
    
    return results

if __name__ == "__main__":
    main()
