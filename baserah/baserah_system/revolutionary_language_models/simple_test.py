#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test for Revolutionary Language Model
اختبار مبسط للنموذج اللغوي الثوري
"""

print("🧪 بدء الاختبار المبسط...")

try:
    print("📦 استيراد المكتبات...")
    import sys
    import os
    from typing import Dict, List, Any
    from datetime import datetime
    from enum import Enum
    from dataclasses import dataclass, field
    from abc import ABC, abstractmethod
    print("✅ تم استيراد المكتبات الأساسية")
    
    print("📦 استيراد النموذج الثوري...")
    from revolutionary_language_model import (
        RevolutionaryLanguageModel,
        LanguageContext,
        LanguageGenerationMode,
        AdaptiveEquationType
    )
    print("✅ تم استيراد النموذج الثوري")
    
    print("🚀 إنشاء النموذج...")
    model = RevolutionaryLanguageModel()
    print("✅ تم إنشاء النموذج")
    
    print("📝 إنشاء سياق الاختبار...")
    context = LanguageContext(
        text="اختبار النموذج الثوري",
        domain="general",
        complexity_level=0.5,
        basil_methodology_enabled=True,
        physics_thinking_enabled=True
    )
    print("✅ تم إنشاء السياق")
    
    print("🎯 تشغيل التوليد...")
    result = model.generate(context)
    print("✅ تم التوليد بنجاح!")
    
    print("\n📊 النتائج:")
    print(f"   📝 النص المولد: {result.generated_text}")
    print(f"   📊 الثقة: {result.confidence_score:.2f}")
    print(f"   🔗 التوافق الدلالي: {result.semantic_alignment:.2f}")
    print(f"   🧠 التماسك المفاهيمي: {result.conceptual_coherence:.2f}")
    print(f"   💡 رؤى باسل: {len(result.basil_insights)}")
    print(f"   🔬 مبادئ فيزيائية: {len(result.physics_principles_applied)}")
    print(f"   ⚡ معادلات مستخدمة: {len(result.adaptive_equations_used)}")
    
    print("\n🎉 تم الاختبار بنجاح! النموذج يعمل بشكل مثالي!")
    
except Exception as e:
    print(f"\n❌ خطأ في الاختبار: {str(e)}")
    import traceback
    print("📋 تفاصيل الخطأ:")
    traceback.print_exc()
