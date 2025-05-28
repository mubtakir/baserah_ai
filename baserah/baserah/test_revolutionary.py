#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'baserah_system/symbolic_processing')

try:
    from revolutionary_expert_explorer_system import RevolutionaryExpertExplorerSystem
    print("🎉 تم استيراد النظام الرمزي الثوري بنجاح!")
    
    # إنشاء النظام الثوري
    revolutionary_system = RevolutionaryExpertExplorerSystem()
    print("✅ تم إنشاء النظام الرمزي الثوري")
    print(f"🧠 معادلات رمزية: {len(revolutionary_system.symbolic_equations)}")
    print(f"🌟 قواعد المعرفة: {len(revolutionary_system.revolutionary_knowledge_bases)}")
    print("🎉 اختبار أساسي مكتمل!")
    
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()
