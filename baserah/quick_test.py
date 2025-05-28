#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Add baserah_system to path
sys.path.insert(0, 'baserah_system')

print("🌟 اختبار سريع للنظام الموحد 🌟")
print("="*50)

# Test 1: Revolutionary Foundation
print("\n1️⃣ اختبار الأساس الثوري...")
try:
    from revolutionary_core.unified_revolutionary_foundation import get_revolutionary_foundation
    foundation = get_revolutionary_foundation()
    print(f"✅ الأساس الثوري: {len(foundation.revolutionary_terms)} حد")
except Exception as e:
    print(f"❌ الأساس الثوري: {e}")

# Test 2: Integration System
print("\n2️⃣ اختبار نظام التكامل...")
try:
    from integration.unified_system_integration import UnifiedSystemIntegration
    integration = UnifiedSystemIntegration()
    print("✅ نظام التكامل: تم إنشاؤه")
except Exception as e:
    print(f"❌ نظام التكامل: {e}")

# Test 3: Revolutionary Learning
print("\n3️⃣ اختبار التعلم الثوري...")
try:
    from learning.reinforcement.innovative_rl_unified import create_unified_revolutionary_learning_system
    learning = create_unified_revolutionary_learning_system()
    decision = learning.make_expert_decision({"complexity": 0.5})
    print(f"✅ التعلم الثوري: {decision.get('decision', 'يعمل')}")
except Exception as e:
    print(f"❌ التعلم الثوري: {e}")

# Test 4: Dream Interpretation
print("\n4️⃣ اختبار تفسير الأحلام...")
try:
    from dream_interpretation.revolutionary_dream_interpreter_unified import create_unified_revolutionary_dream_interpreter
    interpreter = create_unified_revolutionary_dream_interpreter()
    print("✅ تفسير الأحلام: تم إنشاؤه")
except Exception as e:
    print(f"❌ تفسير الأحلام: {e}")

# Test 5: Web Interface
print("\n5️⃣ اختبار واجهة الويب...")
try:
    from interfaces.web.unified_web_interface import create_unified_web_interface
    web = create_unified_web_interface()
    print("✅ واجهة الويب: تم إنشاؤها")
except Exception as e:
    print(f"❌ واجهة الويب: {e}")

# Test 6: Desktop Interface
print("\n6️⃣ اختبار واجهة سطح المكتب...")
try:
    from interfaces.desktop.unified_desktop_interface import create_unified_desktop_interface
    desktop = create_unified_desktop_interface()
    print("✅ واجهة سطح المكتب: تم إنشاؤها")
except Exception as e:
    print(f"❌ واجهة سطح المكتب: {e}")

print("\n🎯 الاختبار السريع مكتمل!")
print("🌟 إبداع باسل يحيى عبدالله محفوظ! 🌟")
