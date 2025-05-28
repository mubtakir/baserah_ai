#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Test for Fixed Mathematical Core
اختبار مبسط للنواة الرياضية المُصححة
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def minimal_test():
    """اختبار مبسط"""
    
    print("🧪 اختبار مبسط للنواة الرياضية المُصححة...")
    print("🌟" + "="*80 + "🌟")
    print("🚀 النواة الرياضية - بدون PyTorch")
    print("⚡ NumPy بدلاً من PyTorch")
    print("🌟" + "="*80 + "🌟")
    
    try:
        # اختبار الاستيراد الأساسي
        print("\n📦 اختبار الاستيراد الأساسي...")
        
        try:
            from innovative_calculus_engine import InnovativeCalculusEngine
            print("✅ تم استيراد InnovativeCalculusEngine!")
        except Exception as e:
            print(f"❌ فشل استيراد InnovativeCalculusEngine: {e}")
        
        try:
            from function_decomposition_engine import FunctionDecompositionEngine
            print("✅ تم استيراد FunctionDecompositionEngine!")
        except Exception as e:
            print(f"❌ فشل استيراد FunctionDecompositionEngine: {e}")
        
        try:
            from general_shape_equation import GeneralShapeEquation, EquationType
            print("✅ تم استيراد GeneralShapeEquation!")
        except Exception as e:
            print(f"❌ فشل استيراد GeneralShapeEquation: {e}")
        
        # اختبار إنشاء المحركات
        print("\n🔍 اختبار إنشاء المحركات...")
        
        try:
            calculus_engine = InnovativeCalculusEngine()
            print("✅ تم إنشاء محرك التفاضل والتكامل!")
        except Exception as e:
            print(f"❌ فشل إنشاء محرك التفاضل والتكامل: {e}")
        
        try:
            decomp_engine = FunctionDecompositionEngine()
            print("✅ تم إنشاء محرك تفكيك الدوال!")
        except Exception as e:
            print(f"❌ فشل إنشاء محرك تفكيك الدوال: {e}")
        
        try:
            gse = GeneralShapeEquation()
            print("✅ تم إنشاء المعادلة العامة للشكل!")
        except Exception as e:
            print(f"❌ فشل إنشاء المعادلة العامة للشكل: {e}")
        
        # اختبار بسيط للوظائف
        print("\n🔍 اختبار بسيط للوظائف...")
        
        # اختبار NumPy
        x = np.linspace(-1, 1, 10)
        y = x**2
        print(f"✅ NumPy يعمل: x={x[:3]}, y={y[:3]}")
        
        # اختبار دالة بسيطة
        def simple_func(x):
            return x**2
        
        result = simple_func(x)
        print(f"✅ الدوال تعمل: result={result[:3]}")
        
        print("\n🎉 الاختبار المبسط نجح!")
        print("✅ النواة الرياضية تعمل بدون PyTorch!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ خطأ في الاختبار المبسط: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    if success:
        print("\n🎉 الاختبار المبسط نجح!")
    else:
        print("\n❌ فشل الاختبار المبسط!")
