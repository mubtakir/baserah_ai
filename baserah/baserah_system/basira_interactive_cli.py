#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basira System Interactive CLI
واجهة سطر الأوامر التفاعلية لنظام بصيرة

Interactive command-line interface for Basira System
showcasing all revolutionary mathematical innovations by Basil Yahya Abdullah.

Author: Basira Development Team
Supervised by: Basil Yahya Abdullah
Version: 3.0.0 - "Interactive CLI"
"""

import sys
import os
import math
from datetime import datetime
from typing import List, Dict, Any

# Import our simple demo components
from basira_simple_demo import SimpleExpertSystem, SimpleGeneralShapeEquation, SimpleInnovativeCalculus, SimpleRevolutionaryDecomposition

class BasiraInteractiveCLI:
    """واجهة سطر الأوامر التفاعلية لنظام بصيرة"""
    
    def __init__(self):
        self.expert_system = SimpleExpertSystem()
        self.running = True
        self.session_history = []
        
        print("🌟" + "="*80 + "🌟")
        print("🚀 نظام بصيرة - الواجهة التفاعلية 🚀")
        print("🚀 Basira System - Interactive Interface 🚀")
        print("🌟 إبداع باسل يحيى عبدالله 🌟")
        print("🌟 Created by Basil Yahya Abdullah 🌟")
        print("🌟" + "="*80 + "🌟")
    
    def show_main_menu(self):
        """عرض القائمة الرئيسية"""
        print("\n📋 القائمة الرئيسية / Main Menu:")
        print("="*50)
        print("1. 📐 اختبار المعادلة العامة للأشكال")
        print("   Test General Shape Equation")
        print("2. 🧮 النظام المبتكر للتفاضل والتكامل")
        print("   Innovative Calculus System")
        print("3. 🌟 النظام الثوري لتفكيك الدوال")
        print("   Revolutionary Function Decomposition")
        print("4. 🎯 عرض توضيحي شامل")
        print("   Comprehensive Demo")
        print("5. 📊 عرض تاريخ الجلسة")
        print("   Show Session History")
        print("6. ℹ️  معلومات عن النظام")
        print("   System Information")
        print("0. 🚪 خروج / Exit")
        print("="*50)
    
    def test_general_equation(self):
        """اختبار المعادلة العامة للأشكال"""
        print("\n📐 اختبار المعادلة العامة للأشكال...")
        print("📐 Testing General Shape Equation...")
        
        print("\nأدخل البيانات للمعالجة:")
        print("Enter data for processing:")
        
        user_input = input("البيانات / Data: ").strip()
        if not user_input:
            user_input = "test_mathematical_function"
        
        result = self.expert_system.general_equation.process(user_input)
        
        print(f"\n✅ نتيجة المعالجة:")
        print(f"✅ Processing Result:")
        print(f"   📊 نوع المعادلة: {result['equation_type']}")
        print(f"   📊 نمط التعلم: {result['learning_mode']}")
        print(f"   📊 الوقت: {result['timestamp']}")
        print(f"   📊 حالة المعالجة: {'نجح' if result['processed'] else 'فشل'}")
        
        self.session_history.append(("General Equation Test", result))
        
        input("\nاضغط Enter للمتابعة...")
    
    def test_innovative_calculus(self):
        """اختبار النظام المبتكر للتفاضل والتكامل"""
        print("\n🧮 النظام المبتكر للتفاضل والتكامل...")
        print("🧮 Innovative Calculus System...")
        print("💡 المفهوم: تكامل = V × A، تفاضل = D × A")
        print("💡 Concept: Integration = V × A, Differentiation = D × A")
        
        print("\nاختر نوع الاختبار:")
        print("Choose test type:")
        print("1. دالة خطية (Linear function)")
        print("2. دالة تربيعية (Quadratic function)")
        print("3. دالة تكعيبية (Cubic function)")
        print("4. إدخال مخصص (Custom input)")
        
        choice = input("الاختيار / Choice (1-4): ").strip()
        
        if choice == "1":
            # دالة خطية: f(x) = 2x + 1
            x_vals = list(range(1, 6))
            f_vals = [2*x + 1 for x in x_vals]
            D_coeffs = [2.0] * len(f_vals)  # مشتقة الدالة الخطية
            V_coeffs = [x + 0.5 for x in x_vals]  # تكامل تقريبي
            func_name = "Linear: f(x) = 2x + 1"
            
        elif choice == "2":
            # دالة تربيعية: f(x) = x^2
            x_vals = list(range(1, 6))
            f_vals = [x**2 for x in x_vals]
            D_coeffs = [2*x for x in x_vals]  # مشتقة x^2 = 2x
            V_coeffs = [x**3/3 for x in x_vals]  # تكامل x^2 = x^3/3
            func_name = "Quadratic: f(x) = x²"
            
        elif choice == "3":
            # دالة تكعيبية: f(x) = x^3
            x_vals = list(range(1, 6))
            f_vals = [x**3 for x in x_vals]
            D_coeffs = [3*x**2 for x in x_vals]  # مشتقة x^3 = 3x^2
            V_coeffs = [x**4/4 for x in x_vals]  # تكامل x^3 = x^4/4
            func_name = "Cubic: f(x) = x³"
            
        else:
            # إدخال مخصص
            print("أدخل قيم الدالة (مفصولة بفواصل):")
            f_input = input("Function values: ").strip()
            try:
                f_vals = [float(x) for x in f_input.split(',')]
                x_vals = list(range(1, len(f_vals) + 1))
                D_coeffs = [1.0] * len(f_vals)  # معاملات افتراضية
                V_coeffs = [1.0] * len(f_vals)  # معاملات افتراضية
                func_name = "Custom function"
            except:
                print("❌ خطأ في الإدخال، سيتم استخدام القيم الافتراضية")
                f_vals = [1, 4, 9, 16, 25]
                x_vals = [1, 2, 3, 4, 5]
                D_coeffs = [2, 4, 6, 8, 10]
                V_coeffs = [0.33, 1.33, 3, 5.33, 8.33]
                func_name = "Default: f(x) = x²"
        
        # إضافة حالة المعاملات
        state = self.expert_system.calculus_engine.add_coefficient_state(f_vals, D_coeffs, V_coeffs)
        
        # التنبؤ بالتفاضل والتكامل
        result = self.expert_system.calculus_engine.predict_calculus(f_vals)
        
        print(f"\n✅ نتائج {func_name}:")
        print(f"✅ Results for {func_name}:")
        print(f"   📊 قيم الدالة: {f_vals}")
        print(f"   📊 التفاضل المقدر: {[round(x, 2) for x in result['derivative']]}")
        print(f"   📊 التكامل المقدر: {[round(x, 2) for x in result['integral']]}")
        print(f"   📊 الطريقة: {result['method']}")
        
        self.session_history.append(("Innovative Calculus Test", {
            "function": func_name,
            "result": result,
            "state": state
        }))
        
        input("\nاضغط Enter للمتابعة...")
    
    def test_revolutionary_decomposition(self):
        """اختبار النظام الثوري لتفكيك الدوال"""
        print("\n🌟 النظام الثوري لتفكيك الدوال...")
        print("🌟 Revolutionary Function Decomposition...")
        print("💡 الفرضية الثورية: A = x.dA - ∫x.d2A")
        print("💡 Revolutionary Hypothesis: A = x.dA - ∫x.d2A")
        
        print("\nاختر الدالة للتفكيك:")
        print("Choose function to decompose:")
        print("1. دالة تربيعية (Quadratic)")
        print("2. دالة تكعيبية (Cubic)")
        print("3. دالة أسية مبسطة (Simple exponential)")
        print("4. إدخال مخصص (Custom)")
        
        choice = input("الاختيار / Choice (1-4): ").strip()
        
        if choice == "1":
            x_vals = [1, 2, 3, 4, 5]
            f_vals = [x**2 for x in x_vals]
            func_name = "Quadratic: f(x) = x²"
            
        elif choice == "2":
            x_vals = [1, 2, 3, 4, 5]
            f_vals = [x**3 for x in x_vals]
            func_name = "Cubic: f(x) = x³"
            
        elif choice == "3":
            x_vals = [0, 1, 2, 3, 4]
            f_vals = [math.exp(x/2) for x in x_vals]  # e^(x/2) للتبسيط
            func_name = "Exponential: f(x) = e^(x/2)"
            
        else:
            print("أدخل قيم x (مفصولة بفواصل):")
            x_input = input("X values: ").strip()
            print("أدخل قيم f(x) (مفصولة بفواصل):")
            f_input = input("F(x) values: ").strip()
            
            try:
                x_vals = [float(x) for x in x_input.split(',')]
                f_vals = [float(x) for x in f_input.split(',')]
                func_name = "Custom function"
            except:
                print("❌ خطأ في الإدخال، سيتم استخدام القيم الافتراضية")
                x_vals = [1, 2, 3, 4, 5]
                f_vals = [1, 4, 9, 16, 25]
                func_name = "Default: f(x) = x²"
        
        # تنفيذ التفكيك الثوري
        result = self.expert_system.decomposition_engine.decompose_simple_function(
            func_name, x_vals, f_vals
        )
        
        print(f"\n✅ نتائج التفكيك الثوري:")
        print(f"✅ Revolutionary Decomposition Results:")
        print(f"   📊 الدالة: {result['function_name']}")
        print(f"   📊 دقة التفكيك: {result['accuracy']:.4f}")
        print(f"   📊 عدد الحدود: {result['n_terms_used']}")
        print(f"   📊 الطريقة: {result['method']}")
        print(f"   📊 القيم الأصلية: {[round(x, 2) for x in result['original_values']]}")
        print(f"   📊 القيم المعاد بناؤها: {[round(x, 2) for x in result['reconstructed_values'][:5]]}...")
        
        self.session_history.append(("Revolutionary Decomposition Test", result))
        
        input("\nاضغط Enter للمتابعة...")
    
    def comprehensive_demo(self):
        """عرض توضيحي شامل"""
        print("\n🎯 العرض التوضيحي الشامل...")
        print("🎯 Comprehensive Demo...")
        
        print("سيتم تشغيل جميع أنظمة بصيرة...")
        print("Running all Basira systems...")
        
        result = self.expert_system.demonstrate_system()
        
        print(f"\n✅ اكتمل العرض التوضيحي الشامل!")
        print(f"✅ Comprehensive demo completed!")
        
        self.session_history.append(("Comprehensive Demo", result))
        
        input("\nاضغط Enter للمتابعة...")
    
    def show_session_history(self):
        """عرض تاريخ الجلسة"""
        print("\n📊 تاريخ الجلسة...")
        print("📊 Session History...")
        
        if not self.session_history:
            print("لا يوجد تاريخ للجلسة بعد.")
            print("No session history yet.")
        else:
            for i, (test_name, result) in enumerate(self.session_history, 1):
                print(f"\n{i}. {test_name}")
                if isinstance(result, dict) and 'timestamp' in result:
                    print(f"   الوقت: {result['timestamp']}")
                print(f"   النوع: {type(result).__name__}")
        
        input("\nاضغط Enter للمتابعة...")
    
    def show_system_info(self):
        """عرض معلومات النظام"""
        print("\n ℹ️ معلومات نظام بصيرة...")
        print("ℹ️ Basira System Information...")
        
        print("\n🌟 المبدع: باسل يحيى عبدالله من العراق/الموصل")
        print("🌟 Creator: Basil Yahya Abdullah from Iraq/Mosul")
        
        print("\n🧠 المكونات الأساسية:")
        print("🧠 Core Components:")
        print("   📐 المعادلة العامة للأشكال")
        print("   🧮 النظام المبتكر للتفاضل والتكامل")
        print("   🌟 النظام الثوري لتفكيك الدوال")
        print("   🎯 نظام الخبير المتكامل")
        
        print("\n💡 الابتكارات الرياضية:")
        print("💡 Mathematical Innovations:")
        print("   • تكامل = V × A (معاملات التكامل)")
        print("   • تفاضل = D × A (معاملات التفاضل)")
        print("   • A = x.dA - ∫x.d2A (الفرضية الثورية)")
        print("   • المتسلسلة مع الإشارات المتعاقبة")
        
        print(f"\n📅 الإصدار: 3.0.0")
        print(f"📅 Version: 3.0.0")
        print(f"🕐 الوقت الحالي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        input("\nاضغط Enter للمتابعة...")
    
    def run(self):
        """تشغيل الواجهة التفاعلية"""
        while self.running:
            try:
                self.show_main_menu()
                choice = input("\nاختر من القائمة / Choose from menu: ").strip()
                
                if choice == "1":
                    self.test_general_equation()
                elif choice == "2":
                    self.test_innovative_calculus()
                elif choice == "3":
                    self.test_revolutionary_decomposition()
                elif choice == "4":
                    self.comprehensive_demo()
                elif choice == "5":
                    self.show_session_history()
                elif choice == "6":
                    self.show_system_info()
                elif choice == "0":
                    print("\n🚪 شكراً لاستخدام نظام بصيرة!")
                    print("🚪 Thank you for using Basira System!")
                    print("🌟 تحية لباسل يحيى عبدالله!")
                    print("🌟 Salute to Basil Yahya Abdullah!")
                    self.running = False
                else:
                    print("\n❌ اختيار غير صحيح، حاول مرة أخرى")
                    print("❌ Invalid choice, please try again")
                    input("اضغط Enter للمتابعة...")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 تم إيقاف النظام بواسطة المستخدم")
                print("🛑 System stopped by user")
                self.running = False
            except Exception as e:
                print(f"\n❌ خطأ: {e}")
                print(f"❌ Error: {e}")
                input("اضغط Enter للمتابعة...")

def main():
    """الدالة الرئيسية"""
    cli = BasiraInteractiveCLI()
    cli.run()

if __name__ == "__main__":
    main()
