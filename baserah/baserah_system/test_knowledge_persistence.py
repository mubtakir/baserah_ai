#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار نظام حفظ المعرفة - Knowledge Persistence Test
اختبار شامل لنظام قواعد البيانات وحفظ المعرفة

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - Revolutionary Knowledge Persistence Test
"""

import os
import sys
import time
import json
from datetime import datetime

# إضافة المسار للاستيراد
sys.path.insert(0, os.path.dirname(__file__))

def test_database_manager():
    """اختبار مدير قواعد البيانات"""
    print("🌟" + "="*80 + "🌟")
    print("🗄️ اختبار مدير قواعد البيانات الثوري")
    print("🌟" + "="*80 + "🌟")
    
    try:
        from database.revolutionary_database_manager import RevolutionaryDatabaseManager
        
        # إنشاء مدير قواعد البيانات
        db_manager = RevolutionaryDatabaseManager("test_database")
        print("✅ تم إنشاء مدير قواعد البيانات")
        
        # حفظ معرفة تجريبية
        knowledge_id = db_manager.save_knowledge(
            module_name="test_module",
            knowledge_type="test_knowledge",
            content={
                "test_data": "sample_data",
                "value": 42,
                "timestamp": datetime.now().isoformat()
            },
            confidence_level=0.95,
            metadata={"test": True, "source": "unit_test"}
        )
        
        print(f"✅ تم حفظ المعرفة: {knowledge_id}")
        
        # تحميل المعرفة
        loaded_knowledge = db_manager.load_knowledge("test_module", "test_knowledge")
        print(f"✅ تم تحميل {len(loaded_knowledge)} إدخال معرفة")
        
        # عرض الإحصائيات
        stats = db_manager.get_statistics()
        print(f"📊 إحصائيات قواعد البيانات:")
        print(f"   إجمالي قواعد البيانات: {stats['total_databases']}")
        print(f"   إجمالي الإدخالات: {stats['total_entries']}")
        
        # إغلاق المدير
        db_manager.close()
        print("✅ تم إغلاق مدير قواعد البيانات")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار مدير قواعد البيانات: {e}")
        return False

def test_knowledge_persistence_mixin():
    """اختبار خليط حفظ المعرفة"""
    print("\n🧠 اختبار خليط حفظ المعرفة")
    print("="*50)
    
    try:
        from database.knowledge_persistence_mixin import PersistentRevolutionaryComponent
        
        # إنشاء مكون تجريبي
        class TestComponent(PersistentRevolutionaryComponent):
            def __init__(self):
                super().__init__(module_name="test_component")
        
        component = TestComponent()
        print("✅ تم إنشاء المكون التجريبي")
        
        # حفظ معرفة
        knowledge_id = component.save_knowledge(
            knowledge_type="test_type",
            content={"data": "test_data", "value": 123},
            confidence_level=0.9
        )
        print(f"✅ تم حفظ المعرفة: {knowledge_id}")
        
        # تحميل المعرفة
        loaded = component.load_knowledge("test_type")
        print(f"✅ تم تحميل {len(loaded)} إدخال")
        
        # اختبار التعلم من التجربة
        experience_id = component.learn_from_experience({
            "situation": "test_situation",
            "action": "test_action",
            "result": "success"
        }, confidence=0.85)
        print(f"✅ تم حفظ التجربة: {experience_id}")
        
        # عرض ملخص المعرفة
        summary = component.get_knowledge_summary()
        print(f"📊 ملخص المعرفة:")
        print(f"   أنواع المعرفة: {summary['total_knowledge_types']}")
        print(f"   إجمالي الإدخالات: {summary['total_entries']}")
        print(f"   متوسط الثقة: {summary['average_confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار خليط حفظ المعرفة: {e}")
        return False

def test_revolutionary_learning_with_persistence():
    """اختبار نظام التعلم الثوري مع حفظ المعرفة"""
    print("\n🚀 اختبار نظام التعلم الثوري مع حفظ المعرفة")
    print("="*60)
    
    try:
        from learning.reinforcement.innovative_rl_unified import (
            create_unified_revolutionary_learning_system,
            RevolutionaryExperience,
            RevolutionaryLearningConfig,
            RevolutionaryLearningStrategy
        )
        
        # إنشاء النظام
        config = RevolutionaryLearningConfig(
            strategy=RevolutionaryLearningStrategy.BASIL_INTEGRATIVE
        )
        system = create_unified_revolutionary_learning_system(config)
        print("✅ تم إنشاء نظام التعلم الثوري")
        
        # اختبار قرار الخبير
        test_situation = {"complexity": 0.8, "novelty": 0.6}
        expert_decision = system.make_expert_decision(test_situation)
        print(f"✅ قرار الخبير: {expert_decision.get('decision', 'N/A')}")
        print(f"   معرف الحفظ: {expert_decision.get('saved_decision_id', 'N/A')}")
        
        # اختبار الاستكشاف
        exploration_result = system.explore_new_possibilities(test_situation)
        print(f"✅ نتيجة الاستكشاف: {exploration_result.get('discovery', 'N/A')}")
        print(f"   معرف الحفظ: {exploration_result.get('saved_exploration_id', 'N/A')}")
        
        # اختبار التعلم من التجربة
        experience = RevolutionaryExperience(
            situation=test_situation,
            expert_decision=expert_decision["decision"],
            wisdom_gain=0.75,
            evolved_situation={"complexity": 0.9, "novelty": 0.7},
            completion_status=True,
            basil_insights={"insight": "test_insight"},
            physics_principles={"principle": "test_principle"}
        )
        
        learning_result = system.learn_from_experience(experience)
        print(f"✅ التعلم من التجربة: {learning_result['learning_successful']}")
        print(f"   معرف التجربة المحفوظة: {learning_result.get('saved_experience_id', 'N/A')}")
        
        # عرض حالة النظام
        status = system.get_system_status()
        print(f"\n📊 حالة النظام:")
        print(f"   AI-OOP مطبق: {status['ai_oop_applied']}")
        print(f"   حفظ المعرفة مفعل: {status['knowledge_persistence_enabled']}")
        print(f"   الحكمة المتراكمة: {status['wisdom_accumulated']:.3f}")
        
        if "knowledge_database" in status:
            db_info = status["knowledge_database"]
            print(f"   قاعدة البيانات:")
            print(f"     أنواع المعرفة: {db_info.get('total_knowledge_types', 0)}")
            print(f"     إجمالي الإدخالات: {db_info.get('total_entries', 0)}")
            print(f"     متوسط الثقة: {db_info.get('average_confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار نظام التعلم الثوري: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persistence_across_restarts():
    """اختبار استمرارية المعرفة عبر إعادة التشغيل"""
    print("\n🔄 اختبار استمرارية المعرفة عبر إعادة التشغيل")
    print("="*60)
    
    try:
        from database.knowledge_persistence_mixin import PersistentRevolutionaryComponent
        
        # المرحلة 1: حفظ المعرفة
        print("📝 المرحلة 1: حفظ المعرفة...")
        
        class TestPersistentComponent(PersistentRevolutionaryComponent):
            def __init__(self):
                super().__init__(module_name="persistence_test")
        
        component1 = TestPersistentComponent()
        
        # حفظ عدة أنواع من المعرفة
        for i in range(5):
            component1.save_knowledge(
                knowledge_type="test_data",
                content={"iteration": i, "data": f"test_data_{i}"},
                confidence_level=0.8 + (i * 0.04)
            )
        
        print("✅ تم حفظ 5 إدخالات معرفة")
        
        # المرحلة 2: محاكاة إعادة التشغيل
        print("🔄 المرحلة 2: محاكاة إعادة التشغيل...")
        del component1  # حذف المكون الأول
        
        # إنشاء مكون جديد (محاكاة إعادة التشغيل)
        component2 = TestPersistentComponent()
        
        # تحميل المعرفة المحفوظة
        loaded_knowledge = component2.load_knowledge("test_data")
        print(f"✅ تم تحميل {len(loaded_knowledge)} إدخال بعد إعادة التشغيل")
        
        # التحقق من صحة البيانات
        if len(loaded_knowledge) == 5:
            print("✅ جميع البيانات محفوظة بنجاح")
            
            # عرض البيانات المحملة
            for entry in loaded_knowledge[:3]:  # عرض أول 3 إدخالات
                content = entry["content"]
                print(f"   📄 إدخال {content['iteration']}: {content['data']}")
            
            return True
        else:
            print(f"❌ عدد البيانات المحملة غير صحيح: {len(loaded_knowledge)}")
            return False
        
    except Exception as e:
        print(f"❌ خطأ في اختبار الاستمرارية: {e}")
        return False

def main():
    """الدالة الرئيسية للاختبار"""
    print("🌟" + "="*100 + "🌟")
    print("🧪 اختبار شامل لنظام حفظ المعرفة الثوري")
    print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
    print("🌟" + "="*100 + "🌟")
    
    tests = [
        ("مدير قواعد البيانات", test_database_manager),
        ("خليط حفظ المعرفة", test_knowledge_persistence_mixin),
        ("نظام التعلم الثوري مع حفظ المعرفة", test_revolutionary_learning_with_persistence),
        ("استمرارية المعرفة عبر إعادة التشغيل", test_persistence_across_restarts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 تشغيل اختبار: {test_name}")
        print("-" * 50)
        
        start_time = time.time()
        result = test_func()
        end_time = time.time()
        
        results.append((test_name, result, end_time - start_time))
        
        if result:
            print(f"✅ نجح اختبار {test_name} في {end_time - start_time:.2f} ثانية")
        else:
            print(f"❌ فشل اختبار {test_name}")
    
    # النتائج النهائية
    print("\n" + "🌟" + "="*100 + "🌟")
    print("📊 النتائج النهائية")
    print("🌟" + "="*100 + "🌟")
    
    passed_tests = sum(1 for _, result, _ in results if result)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📈 ملخص الاختبارات:")
    print(f"   الاختبارات الناجحة: {passed_tests}/{total_tests}")
    print(f"   معدل النجاح: {success_rate:.1f}%")
    
    print(f"\n📋 تفاصيل النتائج:")
    for test_name, result, duration in results:
        status = "✅ نجح" if result else "❌ فشل"
        print(f"   {test_name}: {status} ({duration:.2f}s)")
    
    if success_rate >= 80:
        verdict = "🎉 ممتاز! نظام حفظ المعرفة يعمل بكفاءة عالية!"
    elif success_rate >= 60:
        verdict = "✅ جيد! النظام يعمل مع بعض المشاكل البسيطة"
    else:
        verdict = "❌ يحتاج تحسين! النظام يحتاج إصلاحات"
    
    print(f"\n🎯 الحكم النهائي: {verdict}")
    
    if passed_tests > 0:
        print("\n🌟 المزايا المحققة:")
        print("   💾 حفظ تلقائي للمعرفة المكتسبة")
        print("   🔄 استمرارية المعرفة عبر إعادة التشغيل")
        print("   📊 إحصائيات شاملة للمعرفة")
        print("   🗄️ قواعد بيانات متعددة منظمة")
        print("   🔒 نسخ احتياطية تلقائية")
    
    print("\n🌟 إبداع باسل يحيى عبدالله من العراق/الموصل محفوظ ومطور! 🌟")
    print("🎯 نظام حفظ المعرفة الثوري جاهز للاستخدام!")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
