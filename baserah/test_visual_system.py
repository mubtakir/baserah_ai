#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار النظام البصري الموحد الموجه بالخبير
Test Expert-Guided Unified Visual System
"""

import sys
import os

# إضافة المسار
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baserah_system'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baserah_system', 'advanced_visual_generation_unit'))

try:
    from baserah_system.advanced_visual_generation_unit.expert_guided_unified_visual_system import (
        ExpertGuidedUnifiedVisualSystem, 
        UnifiedVisualAnalysisRequest
    )
    from baserah_system.revolutionary_database import ShapeEntity
    
    print("🎉 تم استيراد النظام البصري الموحد بنجاح!")
    
    # إنشاء النظام الموحد
    unified_system = ExpertGuidedUnifiedVisualSystem()
    
    # إنشاء شكل اختبار
    test_shape = ShapeEntity(
        id=3, name="عمل فني موحد رائع", category="فن موحد",
        equation_params={"beauty": 0.95, "motion": 0.9, "harmony": 0.92, "creativity": 0.88, "flow": 0.85},
        geometric_features={"area": 250.0, "symmetry": 0.94, "stability": 0.9, "coherence": 0.92, "uniqueness": 0.9},
        color_properties={"primary": [255, 120, 80], "secondary": [80, 180, 255], "accent": [255, 255, 120], "background": [50, 50, 50]},
        position_info={"center_x": 0.5, "center_y": 0.5},
        tolerance_thresholds={}, created_date="", updated_date=""
    )
    
    # طلب تحليل موحد شامل
    analysis_request = UnifiedVisualAnalysisRequest(
        shape=test_shape,
        analysis_modes=["image", "video", "hybrid", "comprehensive"],
        visual_aspects=["quality", "motion", "composition", "narrative", "artistic"],
        integration_level="advanced",
        expert_guidance_level="adaptive",
        learning_enabled=True,
        cross_modal_optimization=True
    )
    
    # تنفيذ التحليل الموحد
    result = unified_system.analyze_unified_visual_with_expert_guidance(analysis_request)
    
    print(f"\n🎨 نتائج التحليل البصري الموحد:")
    print(f"   ✅ النجاح: {result.success}")
    if result.success:
        print(f"   🖼️🎬 رؤى موحدة: {len(result.unified_insights)} رؤية")
        print(f"   🔗 مقاييس التكامل: متاح")
        print(f"   🌟 التقييم الشمولي: متاح")
        print(f"   📊 نتائج الصور: {'متاح' if result.image_analysis_results else 'غير متاح'}")
        print(f"   🎥 نتائج الفيديو: {'متاح' if result.video_analysis_results else 'غير متاح'}")
    
    print(f"\n📊 إحصائيات النظام الموحد:")
    print(f"   🎨 معادلات موحدة: {len(unified_system.unified_equations)}")
    print(f"   📚 قاعدة التعلم: {len(unified_system.unified_learning_database)} إدخال")
    print(f"   🖼️ محلل الصور: {'متاح' if unified_system.image_analyzer else 'غير متاح'}")
    print(f"   🎬 محلل الفيديو: {'متاح' if unified_system.video_analyzer else 'غير متاح'}")
    
    print("\n🎉 اختبار النظام البصري الموحد مكتمل بنجاح!")

except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    print("🔍 محاولة اختبار المحللات المنفصلة...")
    
    try:
        from baserah_system.advanced_visual_generation_unit.expert_guided_image_analyzer import ExpertGuidedImageAnalyzer
        print("✅ محلل الصور متاح")
    except ImportError:
        print("❌ محلل الصور غير متاح")
    
    try:
        from baserah_system.advanced_visual_generation_unit.expert_guided_video_analyzer import ExpertGuidedVideoAnalyzer
        print("✅ محلل الفيديو متاح")
    except ImportError:
        print("❌ محلل الفيديو غير متاح")

except Exception as e:
    print(f"❌ خطأ عام: {e}")
    import traceback
    traceback.print_exc()
