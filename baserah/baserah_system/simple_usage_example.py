#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مثال استخدام بسيط - نظام بصيرة الثوري
Simple Usage Example - Basira Revolutionary System

إبداع باسل يحيى عبدالله من العراق/الموصل
Created by: Basil Yahya Abdullah - Iraq/Mosul

هذا مثال بسيط يوضح كيفية استخدام نظام بصيرة للمبتدئين
This is a simple example showing how to use Basira system for beginners
"""

import sys
import os
from datetime import datetime

def print_welcome():
    """طباعة رسالة الترحيب"""
    print("🌟" + "="*60 + "🌟")
    print("🎨 مرحباً بك في نظام بصيرة الثوري")
    print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل")
    print("🌟" + "="*60 + "🌟")
    print()

def check_requirements():
    """فحص المتطلبات الأساسية"""
    print("🔍 فحص المتطلبات الأساسية...")
    
    required_modules = ['numpy', 'matplotlib', 'PIL']
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'PIL':
                import PIL
            else:
                __import__(module)
            print(f"   ✅ {module}: متاح")
        except ImportError:
            print(f"   ❌ {module}: غير متاح")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ المكتبات المفقودة: {', '.join(missing_modules)}")
        print("💡 لتثبيتها: pip install numpy matplotlib pillow")
        return False
    
    print("✅ جميع المتطلبات متوفرة!")
    return True

def simple_database_example():
    """مثال بسيط لاستخدام قاعدة البيانات"""
    print("\n📊 مثال 1: استخدام قاعدة البيانات الثورية")
    print("-" * 50)
    
    try:
        from revolutionary_database import RevolutionaryShapeDatabase
        
        # إنشاء قاعدة البيانات
        db = RevolutionaryShapeDatabase()
        print("✅ تم إنشاء قاعدة البيانات")
        
        # الحصول على جميع الأشكال
        shapes = db.get_all_shapes()
        print(f"📦 عدد الأشكال المتاحة: {len(shapes)}")
        
        # عرض أول 3 أشكال
        print("\n🎯 الأشكال المتاحة:")
        for i, shape in enumerate(shapes[:3], 1):
            print(f"   {i}. {shape.name} ({shape.category})")
        
        if shapes:
            # اختيار أول شكل للاختبار
            selected_shape = shapes[0]
            print(f"\n🎯 الشكل المختار للاختبار: {selected_shape.name}")
            return selected_shape
        else:
            print("⚠️ لا توجد أشكال في قاعدة البيانات")
            return None
            
    except Exception as e:
        print(f"❌ خطأ في قاعدة البيانات: {e}")
        return None

def simple_visual_generation_example(shape):
    """مثال بسيط للتوليد البصري"""
    print("\n🎨 مثال 2: التوليد البصري البسيط")
    print("-" * 50)
    
    if not shape:
        print("⚠️ لا يوجد شكل للاختبار")
        return
    
    try:
        from advanced_visual_generation_unit import (
            ComprehensiveVisualSystem, 
            ComprehensiveVisualRequest
        )
        
        # إنشاء النظام البصري
        print("🔄 إنشاء النظام البصري...")
        visual_system = ComprehensiveVisualSystem()
        print("✅ تم إنشاء النظام البصري")
        
        # إنشاء طلب بسيط
        print(f"🎯 إنشاء طلب توليد لـ: {shape.name}")
        request = ComprehensiveVisualRequest(
            shape=shape,
            output_types=["image"],           # صورة فقط
            quality_level="standard",        # جودة عادية للسرعة
            artistic_styles=["digital_art"], # فن رقمي
            physics_simulation=True,         # محاكاة فيزيائية
            expert_analysis=True             # تحليل خبير
        )
        
        # تنفيذ التوليد
        print("🚀 بدء التوليد...")
        start_time = datetime.now()
        
        result = visual_system.create_comprehensive_visual_content(request)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # عرض النتائج
        if result.success:
            print("✅ تم التوليد بنجاح!")
            print(f"⏱️ وقت المعالجة: {processing_time:.2f} ثانية")
            print(f"📁 الملفات المولدة:")
            
            for content_type, file_path in result.generated_content.items():
                quality = result.quality_metrics.get(content_type, 0) * 100
                print(f"   📄 {content_type}: {file_path} (جودة: {quality:.1f}%)")
            
            # عرض التوصيات
            if result.recommendations:
                print(f"\n💡 التوصيات:")
                for i, rec in enumerate(result.recommendations[:3], 1):
                    print(f"   {i}. {rec}")
            
            return True
        else:
            print("❌ فشل التوليد:")
            for error in result.error_messages:
                print(f"   • {error}")
            return False
            
    except Exception as e:
        print(f"❌ خطأ في التوليد البصري: {e}")
        return False

def simple_integrated_analysis_example(shape):
    """مثال بسيط للتحليل المتكامل"""
    print("\n🧠 مثال 3: التحليل المتكامل البسيط")
    print("-" * 50)
    
    if not shape:
        print("⚠️ لا يوجد شكل للاختبار")
        return
    
    try:
        from integrated_drawing_extraction_unit import IntegratedDrawingExtractionUnit
        
        # إنشاء الوحدة المتكاملة
        print("🔄 إنشاء الوحدة المتكاملة...")
        integrated_unit = IntegratedDrawingExtractionUnit()
        print("✅ تم إنشاء الوحدة المتكاملة")
        
        # تنفيذ دورة متكاملة
        print(f"🎯 تنفيذ دورة متكاملة لـ: {shape.name}")
        print("🔄 الدورة: رسم → استنباط → فيزياء → خبير → توازن → تعلم")
        
        cycle_result = integrated_unit.execute_integrated_cycle(shape)
        
        # عرض النتائج
        print(f"\n📊 نتائج التحليل المتكامل:")
        print(f"   ✅ نجاح الدورة: {cycle_result['overall_success']}")
        print(f"   📈 النتيجة الإجمالية: {cycle_result.get('overall_score', 0):.2%}")
        
        # تفاصيل التحليل الفيزيائي
        if 'physics_analysis' in cycle_result:
            physics = cycle_result['physics_analysis']
            print(f"\n🔬 التحليل الفيزيائي:")
            print(f"   📊 الدقة الفيزيائية: {physics.get('physical_accuracy', 0):.2%}")
            print(f"   🎯 نتيجة الواقعية: {physics.get('realism_score', 0):.2%}")
            print(f"   ⚠️ تناقضات مكتشفة: {physics.get('contradiction_detected', False)}")
        
        # تفاصيل التوازن الفني-الفيزيائي
        if 'artistic_physics_balance' in cycle_result:
            balance = cycle_result['artistic_physics_balance']
            print(f"\n🎨 التوازن الفني-الفيزيائي:")
            print(f"   🎨 الجمال الفني: {balance.get('artistic_beauty', 0):.2%}")
            print(f"   🔬 الدقة الفيزيائية: {balance.get('physical_accuracy', 0):.2%}")
            print(f"   🌟 التناغم الإجمالي: {balance.get('overall_harmony', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في التحليل المتكامل: {e}")
        return False

def show_system_statistics():
    """عرض إحصائيات النظام"""
    print("\n📊 إحصائيات النظام")
    print("-" * 50)
    
    try:
        from advanced_visual_generation_unit import ComprehensiveVisualSystem
        
        visual_system = ComprehensiveVisualSystem()
        stats = visual_system.get_system_statistics()
        
        print(f"📈 إجمالي الطلبات: {stats['total_requests']}")
        print(f"✅ التوليدات الناجحة: {stats['successful_generations']}")
        print(f"📊 معدل النجاح: {stats.get('success_rate', 0):.1f}%")
        print(f"⏱️ إجمالي وقت المعالجة: {stats['total_processing_time']:.2f} ثانية")
        
        if stats['total_requests'] > 0:
            print(f"⏱️ متوسط وقت المعالجة: {stats['average_processing_time']:.2f} ثانية")
        
        print(f"\n🔧 حالة المكونات:")
        components = stats['components_status']
        for component, status in components.items():
            status_icon = "✅" if status == "متاح" else "⚠️"
            print(f"   {status_icon} {component}: {status}")
            
    except Exception as e:
        print(f"❌ خطأ في جلب الإحصائيات: {e}")

def main():
    """الدالة الرئيسية"""
    print_welcome()
    
    # فحص المتطلبات
    if not check_requirements():
        print("\n❌ لا يمكن المتابعة بدون المتطلبات الأساسية")
        return
    
    print("\n🚀 بدء الأمثلة التوضيحية...")
    
    # مثال 1: قاعدة البيانات
    selected_shape = simple_database_example()
    
    # مثال 2: التوليد البصري
    if selected_shape:
        visual_success = simple_visual_generation_example(selected_shape)
        
        # مثال 3: التحليل المتكامل
        if visual_success:
            simple_integrated_analysis_example(selected_shape)
    
    # عرض الإحصائيات النهائية
    show_system_statistics()
    
    print("\n🎉 انتهت الأمثلة التوضيحية!")
    print("💡 لمزيد من الأمثلة، راجع التوثيق الكامل")
    print("🌟 شكراً لاستخدام نظام بصيرة الثوري!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ تم إيقاف البرنامج بواسطة المستخدم")
    except Exception as e:
        print(f"\n❌ خطأ عام في البرنامج: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 وداعاً!")
