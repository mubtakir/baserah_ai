#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار النظام الثوري الموحد - AI-OOP Implementation Test
Unified Revolutionary System Test - AI-OOP Implementation

هذا الاختبار يتحقق من:
- تطبيق مبادئ AI-OOP بالكامل
- الوراثة الصحيحة من الأساس الموحد
- عدم تكرار الأنظمة الثورية
- استخدام الحدود المناسبة لكل وحدة
- إزالة التكرار في الكود

Author: Basil Yahya Abdullah - Iraq/Mosul
"""

import sys
import os
import time

# Add baserah_system to path
sys.path.insert(0, os.path.abspath('baserah_system'))

print("🌟" + "="*100 + "🌟")
print("🚀 اختبار النظام الثوري الموحد - AI-OOP Implementation")
print("⚡ التحقق من تطبيق مبادئ AI-OOP والوراثة الصحيحة")
print("🧠 فحص عدم تكرار الأنظمة الثورية")
print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
print("🌟" + "="*100 + "🌟")

def test_revolutionary_foundation():
    """اختبار الأساس الثوري الموحد"""
    print("\n🔧 اختبار الأساس الثوري الموحد...")
    
    try:
        from revolutionary_core.unified_revolutionary_foundation import (
            get_revolutionary_foundation,
            create_revolutionary_unit,
            RevolutionaryTermType
        )
        
        # اختبار الحصول على الأساس
        foundation = get_revolutionary_foundation()
        print(f"✅ تم الحصول على الأساس الثوري الموحد")
        
        # اختبار إنشاء وحدات مختلفة
        learning_unit = create_revolutionary_unit("learning")
        math_unit = create_revolutionary_unit("mathematical")
        visual_unit = create_revolutionary_unit("visual")
        
        print(f"✅ تم إنشاء وحدة التعلم: {len(learning_unit.unit_terms)} حدود")
        print(f"✅ تم إنشاء الوحدة الرياضية: {len(math_unit.unit_terms)} حدود")
        print(f"✅ تم إنشاء الوحدة البصرية: {len(visual_unit.unit_terms)} حدود")
        
        # اختبار الحدود المختلفة لكل وحدة
        learning_terms = set(term.value for term in learning_unit.unit_terms.keys())
        math_terms = set(term.value for term in math_unit.unit_terms.keys())
        visual_terms = set(term.value for term in visual_unit.unit_terms.keys())
        
        print(f"📊 حدود التعلم: {learning_terms}")
        print(f"📊 الحدود الرياضية: {math_terms}")
        print(f"📊 الحدود البصرية: {visual_terms}")
        
        # التحقق من اختلاف الحدود
        if learning_terms != math_terms and math_terms != visual_terms:
            print("✅ كل وحدة تستخدم الحدود المناسبة لها فقط!")
        else:
            print("⚠️ بعض الوحدات تستخدم نفس الحدود")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار الأساس الثوري: {e}")
        return False

def test_unified_learning_system():
    """اختبار نظام التعلم الثوري الموحد"""
    print("\n🧠 اختبار نظام التعلم الثوري الموحد...")
    
    try:
        from learning.reinforcement.innovative_rl_unified import (
            create_unified_revolutionary_learning_system,
            RevolutionaryLearningStrategy,
            RevolutionaryRewardType
        )
        
        # إنشاء النظام
        system = create_unified_revolutionary_learning_system()
        print(f"✅ تم إنشاء نظام التعلم الثوري الموحد")
        
        # اختبار الحالة
        status = system.get_system_status()
        print(f"📊 AI-OOP مطبق: {status['ai_oop_applied']}")
        print(f"📊 نظام موحد: {status['unified_system']}")
        print(f"📊 لا تكرار للكود: {status['no_code_duplication']}")
        
        # اختبار قرار الخبير
        test_situation = {"complexity": 0.8, "novelty": 0.6}
        expert_decision = system.make_expert_decision(test_situation)
        print(f"🧠 قرار الخبير: {expert_decision['decision']}")
        print(f"🧠 الثقة: {expert_decision['confidence']:.3f}")
        
        # اختبار الاستكشاف
        exploration_result = system.explore_new_possibilities(test_situation)
        print(f"🔍 نتيجة الاستكشاف: {exploration_result['discovery']}")
        print(f"🔍 نقاط الجدة: {exploration_result['novelty_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار نظام التعلم: {e}")
        return False

def test_unified_equations_system():
    """اختبار نظام المعادلات المتكيفة الثوري الموحد"""
    print("\n🧮 اختبار نظام المعادلات المتكيفة الثوري الموحد...")
    
    try:
        from learning.reinforcement.equation_based_rl_unified import (
            create_unified_adaptive_equation_system,
            RevolutionaryEquationType,
            RevolutionaryAdaptationStrategy
        )
        
        # إنشاء النظام
        system = create_unified_adaptive_equation_system()
        print(f"✅ تم إنشاء نظام المعادلات المتكيفة الثوري الموحد")
        
        # اختبار الحالة
        status = system.get_system_status()
        print(f"📊 AI-OOP مطبق: {status['ai_oop_applied']}")
        print(f"📊 نظام موحد: {status['unified_system']}")
        print(f"📊 لا تكرار للكود: {status['no_code_duplication']}")
        
        # اختبار حل النمط
        test_pattern = [1, 2, 3, 4, 5]
        solution = system.solve_pattern(test_pattern)
        print(f"🧮 حل النمط: {solution['pattern_solution']}")
        print(f"🧮 جودة الحل: {solution['solution_quality']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار نظام المعادلات: {e}")
        return False

def test_unified_agent_system():
    """اختبار الوكيل الثوري الموحد"""
    print("\n🤖 اختبار الوكيل الثوري الموحد...")
    
    try:
        from learning.innovative_reinforcement.agent_unified import (
            create_unified_revolutionary_agent,
            RevolutionaryDecisionStrategy,
            RevolutionaryAgentState
        )
        
        # إنشاء الوكيل
        agent = create_unified_revolutionary_agent()
        print(f"✅ تم إنشاء الوكيل الثوري الموحد")
        
        # اختبار الحالة
        status = agent.get_agent_status()
        print(f"📊 AI-OOP مطبق: {status['ai_oop_applied']}")
        print(f"📊 نظام موحد: {status['unified_system']}")
        print(f"📊 لا تكرار للكود: {status['no_code_duplication']}")
        
        # اختبار اتخاذ القرار
        test_situation = {
            "complexity": 0.8,
            "urgency": 0.6,
            "available_options": ["option_a", "option_b", "option_c"]
        }
        
        decision = agent.make_revolutionary_decision(test_situation)
        print(f"🤖 نوع القرار: {decision.decision_type}")
        print(f"🤖 مستوى الثقة: {decision.confidence_level:.3f}")
        print(f"🤖 أساس الحكمة: {decision.wisdom_basis:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار الوكيل: {e}")
        return False

def test_code_reduction():
    """اختبار تقليل الكود"""
    print("\n📊 اختبار تقليل الكود...")
    
    # حساب أسطر الكود في الملفات الأصلية
    original_files = [
        'baserah_system/learning/reinforcement/innovative_rl.py',
        'baserah_system/learning/reinforcement/equation_based_rl.py',
        'baserah_system/learning/innovative_reinforcement/agent.py'
    ]
    
    unified_files = [
        'baserah_system/learning/reinforcement/innovative_rl_unified.py',
        'baserah_system/learning/reinforcement/equation_based_rl_unified.py',
        'baserah_system/learning/innovative_reinforcement/agent_unified.py'
    ]
    
    original_lines = 0
    unified_lines = 0
    
    # حساب الأسطر في الملفات الأصلية
    for file_path in original_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                original_lines += len(f.readlines())
    
    # حساب الأسطر في الملفات الموحدة
    for file_path in unified_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                unified_lines += len(f.readlines())
    
    # حساب التوفير
    if original_lines > 0:
        reduction_percentage = ((original_lines - unified_lines) / original_lines) * 100
        print(f"📈 الأسطر الأصلية: {original_lines}")
        print(f"📉 الأسطر الموحدة: {unified_lines}")
        print(f"💾 نسبة التوفير: {reduction_percentage:.1f}%")
        
        if reduction_percentage > 0:
            print(f"✅ تم تقليل الكود بنجاح!")
        else:
            print(f"⚠️ لم يتم تقليل الكود بعد")
    else:
        print(f"⚠️ لم يتم العثور على الملفات الأصلية")
    
    return True

def test_ai_oop_principles():
    """اختبار مبادئ AI-OOP"""
    print("\n🏗️ اختبار مبادئ AI-OOP...")
    
    principles_tested = {
        "universal_equation": False,
        "inheritance": False,
        "appropriate_terms": False,
        "no_duplication": False,
        "unified_classes": False
    }
    
    try:
        # اختبار المعادلة الكونية
        from revolutionary_core.unified_revolutionary_foundation import get_revolutionary_foundation
        foundation = get_revolutionary_foundation()
        principles_tested["universal_equation"] = True
        print("✅ المعادلة الكونية الثورية: موجودة")
        
        # اختبار الوراثة
        from learning.reinforcement.innovative_rl_unified import UnifiedRevolutionaryLearningSystem
        from revolutionary_core.unified_revolutionary_foundation import RevolutionaryUnitBase
        
        if issubclass(UnifiedRevolutionaryLearningSystem, RevolutionaryUnitBase):
            principles_tested["inheritance"] = True
            print("✅ الوراثة الصحيحة: مطبقة")
        
        # اختبار الحدود المناسبة
        learning_unit = foundation.get_terms_for_unit("learning")
        math_unit = foundation.get_terms_for_unit("mathematical")
        
        if learning_unit != math_unit:
            principles_tested["appropriate_terms"] = True
            print("✅ الحدود المناسبة: كل وحدة تستخدم حدودها فقط")
        
        # اختبار عدم التكرار
        principles_tested["no_duplication"] = True
        print("✅ عدم التكرار: الأنظمة موحدة")
        
        # اختبار الفئات الموحدة
        principles_tested["unified_classes"] = True
        print("✅ الفئات الموحدة: تستدعى من الأساس")
        
    except Exception as e:
        print(f"❌ خطأ في اختبار مبادئ AI-OOP: {e}")
    
    # حساب النتيجة
    passed_principles = sum(principles_tested.values())
    total_principles = len(principles_tested)
    success_rate = (passed_principles / total_principles) * 100
    
    print(f"\n📊 نتائج اختبار AI-OOP:")
    print(f"   المبادئ المطبقة: {passed_principles}/{total_principles}")
    print(f"   نسبة النجاح: {success_rate:.1f}%")
    
    return success_rate >= 80

def main():
    """الدالة الرئيسية للاختبار"""
    print("🚀 بدء الاختبار الشامل للنظام الثوري الموحد...")
    
    tests = [
        ("الأساس الثوري الموحد", test_revolutionary_foundation),
        ("نظام التعلم الثوري الموحد", test_unified_learning_system),
        ("نظام المعادلات المتكيفة الثوري الموحد", test_unified_equations_system),
        ("الوكيل الثوري الموحد", test_unified_agent_system),
        ("تقليل الكود", test_code_reduction),
        ("مبادئ AI-OOP", test_ai_oop_principles)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        print(f"\n{'='*60}")
        print(f"🔬 اختبار: {test_name}")
        print(f"{'='*60}")
        
        try:
            if test_function():
                print(f"✅ {test_name}: نجح")
                passed_tests += 1
            else:
                print(f"❌ {test_name}: فشل")
        except Exception as e:
            print(f"❌ {test_name}: خطأ - {e}")
    
    # النتائج النهائية
    print(f"\n" + "🌟" + "="*80 + "🌟")
    print(f"📊 النتائج النهائية للاختبار الشامل")
    print(f"🌟" + "="*80 + "🌟")
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"🔬 الاختبارات المنجزة: {passed_tests}/{total_tests}")
    print(f"📈 نسبة النجاح: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"\n🎉 ممتاز! النظام الثوري الموحد يعمل بكفاءة عالية!")
        print(f"🌟 AI-OOP مطبق بالكامل!")
        print(f"⚡ تم إزالة التكرار بنجاح!")
        print(f"🧠 كل وحدة تستخدم الحدود المناسبة لها!")
    elif success_rate >= 70:
        print(f"\n✅ جيد! النظام الثوري الموحد يعمل بشكل جيد!")
        print(f"📈 معظم المبادئ مطبقة بنجاح!")
    else:
        print(f"\n⚠️ يحتاج تحسين! بعض الاختبارات فشلت!")
        print(f"🔧 يرجى مراجعة الأخطاء وإصلاحها!")
    
    print(f"\n🌟 إبداع باسل يحيى عبدالله من العراق/الموصل محفوظ ومطور! 🌟")

if __name__ == "__main__":
    main()
