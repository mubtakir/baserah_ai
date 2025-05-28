#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار النظام الرمزي الثوري للخبير/المستكشف
Test Revolutionary Expert-Explorer System
"""

import sys
import os

# إضافة المسار
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baserah_system', 'symbolic_processing'))

try:
    from revolutionary_expert_explorer_system import (
        RevolutionaryExpertExplorerSystem,
        RevolutionaryExplorationRequest,
        SymbolicIntelligenceLevel,
        ExplorationDimension,
        KnowledgeSynthesisMode
    )
    
    print("🎉 تم استيراد النظام الرمزي الثوري بنجاح!")
    
    # إنشاء النظام الثوري
    revolutionary_system = RevolutionaryExpertExplorerSystem()
    
    # طلب استكشاف ثوري مبسط
    exploration_request = RevolutionaryExplorationRequest(
        target_domain="الذكاء الرمزي المتقدم",
        exploration_dimensions=[
            ExplorationDimension.LOGICAL,
            ExplorationDimension.CREATIVE,
            ExplorationDimension.QUANTUM
        ],
        intelligence_level=SymbolicIntelligenceLevel.REVOLUTIONARY,
        synthesis_mode=KnowledgeSynthesisMode.HOLISTIC,
        objective="تطوير ذكاء رمزي متقدم",
        creative_freedom=0.85,
        quantum_exploration=True,
        transcendence_seeking=True,
        multi_dimensional_analysis=True
    )
    
    print("✅ تم إنشاء طلب الاستكشاف الثوري")
    print("🚀 بدء الاستكشاف...")
    
    # تنفيذ الاستكشاف الثوري
    result = revolutionary_system.explore_with_revolutionary_intelligence(exploration_request)
    
    print(f"\n🧠 نتائج الاستكشاف الثوري:")
    print(f"   ✅ النجاح: {result.success}")
    print(f"   🌟 رؤى مكتشفة: {len(result.discovered_insights)}")
    print(f"   🚀 اختراقات ثورية: {len(result.revolutionary_breakthroughs)}")
    print(f"   🎯 إنجازات التعالي: {len(result.transcendence_achievements)}")
    print(f"   🔮 اكتشافات كمية: {len(result.quantum_discoveries)}")
    print(f"   💡 إبداعات ناشئة: {len(result.creative_innovations)}")
    
    if result.revolutionary_breakthroughs:
        print(f"\n🚀 الاختراقات الثورية:")
        for breakthrough in result.revolutionary_breakthroughs[:2]:
            print(f"   • {breakthrough}")
    
    if result.transcendence_achievements:
        print(f"\n🎯 إنجازات التعالي:")
        for achievement in result.transcendence_achievements[:2]:
            print(f"   • {achievement}")
    
    print(f"\n📊 إحصائيات النظام الثوري:")
    print(f"   🧠 معادلات رمزية: {len(revolutionary_system.symbolic_equations)}")
    print(f"   🌟 قواعد المعرفة: {len(revolutionary_system.revolutionary_knowledge_bases)}")
    print(f"   📚 قاعدة التعلم: {len(revolutionary_system.symbolic_learning_database)} مجال")
    print(f"   🔄 دورات التطور: {revolutionary_system.self_evolution_engine['evolution_cycles']}")
    
    print("\n🎉 اختبار النظام الرمزي الثوري مكتمل بنجاح!")

except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    print("🔍 التحقق من وجود الملفات...")
    
    # فحص وجود الملفات
    revolutionary_path = os.path.join(os.path.dirname(__file__), 'baserah_system', 'symbolic_processing', 'revolutionary_expert_explorer_system.py')
    if os.path.exists(revolutionary_path):
        print("✅ ملف النظام الثوري موجود")
    else:
        print("❌ ملف النظام الثوري غير موجود")

except Exception as e:
    print(f"❌ خطأ عام: {e}")
    import traceback
    traceback.print_exc()
