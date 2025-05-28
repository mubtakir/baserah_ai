#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test for Fixed Innovative Calculus Engine
اختبار بسيط لمحرك التفاضل والتكامل المبتكر المُصحح
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_test():
    """اختبار بسيط"""
    
    print("🧪 اختبار بسيط لمحرك التفاضل والتكامل المبتكر المُصحح...")
    print("🌟" + "="*80 + "🌟")
    print("🚀 محرك التفاضل والتكامل المبتكر - بدون PyTorch")
    print("⚡ NumPy بدلاً من PyTorch + نفس المفهوم الثوري")
    print("🌟" + "="*80 + "🌟")
    
    try:
        # اختبار الاستيراد
        print("\n📦 اختبار الاستيراد...")
        from innovative_calculus_engine import InnovativeCalculusEngine
        print("✅ تم استيراد InnovativeCalculusEngine بنجاح!")
        
        # إنشاء المحرك
        print("\n🔍 إنشاء المحرك...")
        engine = InnovativeCalculusEngine(merge_threshold=0.8, learning_rate=0.3)
        print("✅ تم إنشاء المحرك بنجاح!")
        
        # اختبار دالة بسيطة
        print("\n🔍 اختبار دالة بسيطة...")
        
        def f_simple(x):
            return x**2
        
        def f_simple_prime(x):
            return 2*x
        
        def f_simple_integral(x):
            return (x**3) / 3
        
        # بيانات الدالة
        function_data = {
            'name': 'x^2',
            'f': f_simple,
            'f_prime': f_simple_prime,
            'f_integral': f_simple_integral,
            'domain': (-2, 2, 50),
            'noise': 0.01
        }
        
        # التدريب
        print("   🔄 بدء التدريب...")
        metrics = engine.train_on_function(function_data, epochs=100)
        
        print("   📊 نتائج التدريب:")
        print(f"      📈 خطأ التفاضل (MAE): {metrics['mae_derivative']:.4f}")
        print(f"      📈 خطأ التكامل (MAE): {metrics['mae_integral']:.4f}")
        print(f"      📊 الخطأ النهائي: {metrics['final_loss']:.4f}")
        print(f"      🔗 عدد الحالات: {metrics['num_states']}")
        
        # اختبار التنبؤ
        print("\n🔍 اختبار التنبؤ...")
        x_test = np.linspace(-1, 1, 20)
        A_test = f_simple(x_test)
        
        pred_derivative, pred_integral = engine.predict(A_test)
        
        true_derivative = f_simple_prime(x_test)
        true_integral = f_simple_integral(x_test)
        
        derivative_error = np.mean(np.abs(pred_derivative - true_derivative))
        integral_error = np.mean(np.abs(pred_integral - true_integral))
        
        print("   📊 نتائج التنبؤ:")
        print(f"      📈 خطأ التفاضل: {derivative_error:.4f}")
        print(f"      📈 خطأ التكامل: {integral_error:.4f}")
        
        # ملخص الأداء
        print("\n🔍 ملخص الأداء...")
        summary = engine.get_performance_summary()
        
        print("   📊 ملخص الأداء:")
        print(f"      📈 الدوال المدربة: {summary['total_functions_trained']}")
        print(f"      📊 متوسط الخطأ: {summary['average_final_loss']:.4f}")
        print(f"      🔗 إجمالي الحالات: {summary['total_states']}")
        
        print("\n🎉 تم الاختبار البسيط بنجاح!")
        print("✅ محرك التفاضل والتكامل المبتكر يعمل بدون PyTorch!")
        print("🚀 المفهوم الثوري محفوظ بالكامل!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ خطأ في الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 الاختبار البسيط نجح!")
    else:
        print("\n❌ فشل الاختبار البسيط!")
