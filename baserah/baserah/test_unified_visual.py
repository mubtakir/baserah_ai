#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'baserah_system')

try:
    from baserah_system.revolutionary_database import ShapeEntity
    print("✅ تم استيراد قاعدة البيانات")
    
    # إنشاء شكل اختبار
    test_shape = ShapeEntity(
        id=3, name="عمل فني موحد رائع", category="فن موحد",
        equation_params={"beauty": 0.95, "motion": 0.9, "harmony": 0.92},
        geometric_features={"area": 250.0, "symmetry": 0.94, "stability": 0.9},
        color_properties={"primary": [255, 120, 80], "secondary": [80, 180, 255]},
        position_info={"center_x": 0.5, "center_y": 0.5},
        tolerance_thresholds={}, created_date="", updated_date=""
    )
    print(f"✅ تم إنشاء الشكل: {test_shape.name}")
    
    print("🎉 اختبار أساسي مكتمل!")
    
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()
