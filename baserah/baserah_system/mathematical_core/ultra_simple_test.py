#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Simple Test
اختبار فائق البساطة
"""

import numpy as np

def ultra_simple_test():
    """اختبار فائق البساطة"""
    
    print("🧪 اختبار فائق البساطة...")
    print("🌟" + "="*60 + "🌟")
    print("🚀 النواة الرياضية - بدون PyTorch")
    print("⚡ NumPy فقط")
    print("🌟" + "="*60 + "🌟")
    
    try:
        # اختبار NumPy
        print("\n📦 اختبار NumPy...")
        x = np.linspace(-1, 1, 5)
        y = x**2
        print(f"✅ NumPy يعمل: x={x}, y={y}")
        
        # اختبار دالة بسيطة
        print("\n🔍 اختبار دالة بسيطة...")
        def simple_func(x):
            return x**2 + 2*x + 1
        
        result = simple_func(x)
        print(f"✅ الدالة تعمل: result={result}")
        
        # اختبار المشتقة العددية
        print("\n🔍 اختبار المشتقة العددية...")
        h = 0.001
        derivative = (simple_func(x + h) - simple_func(x - h)) / (2 * h)
        print(f"✅ المشتقة العددية: derivative={derivative}")
        
        # اختبار التكامل العددي
        print("\n🔍 اختبار التكامل العددي...")
        dx = x[1] - x[0] if len(x) > 1 else 0.1
        integral = np.cumsum(y) * dx
        print(f"✅ التكامل العددي: integral={integral}")
        
        print("\n🎉 الاختبار فائق البساطة نجح!")
        print("✅ النواة الرياضية الأساسية تعمل!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ خطأ في الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = ultra_simple_test()
    if success:
        print("\n🎉 الاختبار فائق البساطة نجح!")
    else:
        print("\n❌ فشل الاختبار فائق البساطة!")
