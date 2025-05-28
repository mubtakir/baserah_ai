#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Guided Adaptive Equations Demo for Basira System
عرض توضيحي للمعادلات المتكيفة الموجهة بالخبير - نظام بصيرة

Demonstrates how the Expert/Explorer guides mathematical equation adaptation
in the revolutionary Basira system.

يوضح كيف يقود الخبير/المستكشف تكيف المعادلات الرياضية
في نظام بصيرة الثوري.

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Any

# استيراد النظام المتكامل
try:
    from .integrated_expert_equation_bridge import IntegratedExpertEquationBridge
    from .expert_guided_adaptive_equations import DrawingExtractionAnalysis
except ImportError:
    # للتشغيل المباشر
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from integrated_expert_equation_bridge import IntegratedExpertEquationBridge
    from expert_guided_adaptive_equations import DrawingExtractionAnalysis

# محاكاة ShapeEntity إذا لم تكن متاحة
class MockShapeEntity:
    def __init__(self, name: str, category: str, complexity: int, properties: Dict[str, Any]):
        self.name = name
        self.category = category
        self.complexity = complexity
        self.properties = properties

def print_header(title: str):
    """طباعة عنوان مميز"""
    print("\n" + "🌟" + "="*70 + "🌟")
    print(f"🎯 {title}")
    print("🌟" + "="*70 + "🌟")

def print_section(title: str):
    """طباعة قسم فرعي"""
    print(f"\n📋 {title}")
    print("-" * 50)

def demonstrate_expert_guided_adaptation():
    """عرض توضيحي شامل للتكيف الموجه بالخبير"""
    
    print_header("عرض توضيحي: المعادلات المتكيفة الموجهة بالخبير")
    print("💡 مفهوم ثوري: الخبير/المستكشف يقود تكيف المعادلات الرياضية")
    print("🧠 بدلاً من التكيف العشوائي، نحصل على تكيف ذكي موجه")
    
    # إنشاء النظام المتكامل
    print_section("تهيئة النظام المتكامل")
    bridge = IntegratedExpertEquationBridge()
    
    # إنشاء أشكال متنوعة للاختبار
    test_shapes = [
        MockShapeEntity("دائرة_بسيطة", "هندسي", 3, {"radius": 5, "color": "أحمر"}),
        MockShapeEntity("مثلث_معقد", "هندسي", 8, {"sides": 3, "angles": [60, 60, 60], "style": "فني"}),
        MockShapeEntity("وردة_فنية", "طبيعي", 12, {"petals": 8, "color": "وردي", "texture": "ناعم"}),
        MockShapeEntity("مبنى_معماري", "معماري", 15, {"floors": 10, "style": "حديث", "materials": ["زجاج", "فولاذ"]})
    ]
    
    # متغيرات لتتبع الأداء
    performance_history = []
    adaptation_details = []
    
    print_section("دورات التكيف الموجه بالخبير")
    
    for i, shape in enumerate(test_shapes, 1):
        print(f"\n🔄 الدورة {i}: معالجة {shape.name}")
        print(f"   📊 التعقيد: {shape.complexity}")
        print(f"   🏷️ الفئة: {shape.category}")
        print(f"   📝 الخصائص: {len(shape.properties)}")
        
        # تنفيذ دورة التكيف المتكاملة
        result = bridge.execute_integrated_adaptation_cycle(shape)
        
        # عرض النتائج
        if result.success:
            print(f"   ✅ نجح التكيف!")
            print(f"   📈 التحسن: {result.performance_improvement:.2%}")
            print(f"   🎯 التوصيات: {len(result.recommendations)}")
            
            # عرض أهم التوصيات
            for j, rec in enumerate(result.recommendations[:2], 1):
                print(f"      {j}. {rec}")
            
            # حفظ البيانات للتحليل
            performance_history.append({
                'cycle': i,
                'shape_name': shape.name,
                'complexity': shape.complexity,
                'improvement': result.performance_improvement,
                'success': result.success
            })
            
            adaptation_details.append({
                'shape': shape.name,
                'adaptations': result.equation_adaptations,
                'expert_analysis': result.expert_analysis
            })
            
        else:
            print(f"   ❌ فشل التكيف")
            performance_history.append({
                'cycle': i,
                'shape_name': shape.name,
                'complexity': shape.complexity,
                'improvement': 0.0,
                'success': False
            })
    
    # تحليل النتائج الإجمالية
    print_section("تحليل النتائج الإجمالية")
    
    successful_cycles = [p for p in performance_history if p['success']]
    if successful_cycles:
        avg_improvement = np.mean([p['improvement'] for p in successful_cycles])
        max_improvement = max([p['improvement'] for p in successful_cycles])
        min_improvement = min([p['improvement'] for p in successful_cycles])
        
        print(f"📊 إحصائيات الأداء:")
        print(f"   ✅ الدورات الناجحة: {len(successful_cycles)}/{len(test_shapes)}")
        print(f"   📈 متوسط التحسن: {avg_improvement:.2%}")
        print(f"   🔝 أفضل تحسن: {max_improvement:.2%}")
        print(f"   📉 أقل تحسن: {min_improvement:.2%}")
        
        # تحليل العلاقة بين التعقيد والتحسن
        complexities = [p['complexity'] for p in successful_cycles]
        improvements = [p['improvement'] for p in successful_cycles]
        
        if len(complexities) > 1:
            correlation = np.corrcoef(complexities, improvements)[0, 1]
            print(f"   🔗 الارتباط بين التعقيد والتحسن: {correlation:.3f}")
    
    # عرض تفاصيل التكيفات
    print_section("تفاصيل تكيف المعادلات")
    
    for detail in adaptation_details:
        print(f"\n🧮 {detail['shape']}:")
        
        for eq_name, eq_info in detail['adaptations'].items():
            if eq_info:
                complexity = eq_info.get('current_complexity', 'غير محدد')
                adaptations = eq_info.get('total_adaptations', 0)
                avg_improvement = eq_info.get('average_improvement', 0.0)
                
                print(f"   📐 {eq_name}:")
                print(f"      🔢 التعقيد الحالي: {complexity}")
                print(f"      🔄 عدد التكيفات: {adaptations}")
                print(f"      📈 متوسط التحسن: {avg_improvement:.3f}")
    
    # رسم بياني للأداء
    create_performance_visualization(performance_history)
    
    # تحليل أولويات الدوال الرياضية
    analyze_function_priorities(bridge)
    
    # توصيات للتطوير المستقبلي
    print_section("توصيات للتطوير المستقبلي")
    
    future_recommendations = generate_future_recommendations(performance_history, adaptation_details)
    for i, rec in enumerate(future_recommendations, 1):
        print(f"   {i}. {rec}")
    
    return bridge, performance_history, adaptation_details

def create_performance_visualization(performance_history: List[Dict]):
    """إنشاء رسم بياني للأداء"""
    
    try:
        cycles = [p['cycle'] for p in performance_history]
        improvements = [p['improvement'] for p in performance_history]
        complexities = [p['complexity'] for p in performance_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # الرسم الأول: التحسن عبر الدورات
        colors = ['green' if p['success'] else 'red' for p in performance_history]
        ax1.bar(cycles, improvements, color=colors, alpha=0.7)
        ax1.set_title('تحسن الأداء عبر دورات التكيف الموجه بالخبير')
        ax1.set_xlabel('رقم الدورة')
        ax1.set_ylabel('نسبة التحسن')
        ax1.grid(True, alpha=0.3)
        
        # إضافة تسميات الأشكال
        for i, p in enumerate(performance_history):
            ax1.text(p['cycle'], p['improvement'] + 0.01, p['shape_name'], 
                    rotation=45, ha='left', fontsize=8)
        
        # الرسم الثاني: العلاقة بين التعقيد والتحسن
        successful_points = [(p['complexity'], p['improvement']) for p in performance_history if p['success']]
        if successful_points:
            complexities_success, improvements_success = zip(*successful_points)
            ax2.scatter(complexities_success, improvements_success, color='blue', alpha=0.7, s=100)
            
            # خط الاتجاه
            if len(successful_points) > 1:
                z = np.polyfit(complexities_success, improvements_success, 1)
                p = np.poly1d(z)
                ax2.plot(complexities_success, p(complexities_success), "r--", alpha=0.8)
        
        ax2.set_title('العلاقة بين تعقيد الشكل ونسبة التحسن')
        ax2.set_xlabel('تعقيد الشكل')
        ax2.set_ylabel('نسبة التحسن')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('expert_guided_adaptation_performance.png', dpi=300, bbox_inches='tight')
        print("📊 تم حفظ الرسم البياني: expert_guided_adaptation_performance.png")
        
    except Exception as e:
        print(f"⚠️ تعذر إنشاء الرسم البياني: {e}")

def analyze_function_priorities(bridge: IntegratedExpertEquationBridge):
    """تحليل أولويات الدوال الرياضية"""
    
    print_section("تحليل أولويات الدوال الرياضية")
    
    equations = [
        ('الرسم الفني', bridge.drawing_equation),
        ('الاستنباط الذكي', bridge.extraction_equation),
        ('التحليل الفيزيائي', bridge.physics_equation)
    ]
    
    for eq_name, equation in equations:
        if equation:
            print(f"\n🧮 {eq_name}:")
            
            # الحصول على أوزان الدوال
            function_weights = equation.function_importance_weights.detach().cpu().numpy()
            function_names = equation.function_names
            
            # ترتيب الدوال حسب الأهمية
            sorted_indices = np.argsort(function_weights)[::-1]
            
            print("   📊 ترتيب الدوال حسب الأهمية:")
            for i, idx in enumerate(sorted_indices[:5], 1):
                func_name = function_names[idx]
                weight = function_weights[idx]
                print(f"      {i}. {func_name}: {weight:.4f}")

def generate_future_recommendations(performance_history: List[Dict], 
                                  adaptation_details: List[Dict]) -> List[str]:
    """توليد توصيات للتطوير المستقبلي"""
    
    recommendations = []
    
    # تحليل معدل النجاح
    success_rate = sum(1 for p in performance_history if p['success']) / len(performance_history)
    
    if success_rate < 0.7:
        recommendations.append("تحسين خوارزميات تحليل الخبير لزيادة معدل النجاح")
    
    # تحليل التحسن
    successful_improvements = [p['improvement'] for p in performance_history if p['success']]
    if successful_improvements:
        avg_improvement = np.mean(successful_improvements)
        
        if avg_improvement < 0.3:
            recommendations.append("زيادة قوة التكيف في المعادلات الرياضية")
        elif avg_improvement > 0.8:
            recommendations.append("الحفاظ على الإعدادات الحالية وتوسيع نطاق الاختبار")
    
    # تحليل التعقيد
    high_complexity_shapes = [p for p in performance_history if p['complexity'] > 10]
    if high_complexity_shapes:
        high_complexity_success = sum(1 for p in high_complexity_shapes if p['success'])
        if high_complexity_success / len(high_complexity_shapes) < 0.5:
            recommendations.append("تطوير استراتيجيات خاصة للأشكال عالية التعقيد")
    
    # توصيات عامة
    recommendations.extend([
        "إضافة المزيد من الدوال الرياضية المتخصصة",
        "تطوير نظام تعلم تراكمي للخبير",
        "إنشاء قاعدة بيانات للتكيفات الناجحة",
        "تطوير واجهة مرئية لمراقبة التكيف في الوقت الفعلي"
    ])
    
    return recommendations

def main():
    """الدالة الرئيسية للعرض التوضيحي"""
    
    print_header("نظام بصيرة: المعادلات المتكيفة الموجهة بالخبير")
    print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل")
    print("💡 مفهوم ثوري: الخبير يقود التكيف بدلاً من العشوائية")
    
    try:
        # تشغيل العرض التوضيحي
        bridge, performance_history, adaptation_details = demonstrate_expert_guided_adaptation()
        
        # حفظ النتائج
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_history': performance_history,
            'summary': {
                'total_cycles': len(performance_history),
                'successful_cycles': sum(1 for p in performance_history if p['success']),
                'average_improvement': np.mean([p['improvement'] for p in performance_history if p['success']]) if any(p['success'] for p in performance_history) else 0.0
            }
        }
        
        with open('expert_guided_adaptation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print_section("ملخص النتائج")
        print(f"✅ تم إنجاز {results['summary']['total_cycles']} دورة تكيف")
        print(f"🎯 نجح منها {results['summary']['successful_cycles']} دورة")
        print(f"📈 متوسط التحسن: {results['summary']['average_improvement']:.2%}")
        print("💾 تم حفظ النتائج في: expert_guided_adaptation_results.json")
        
        print_header("انتهى العرض التوضيحي بنجاح!")
        print("🌟 الخبير/المستكشف يقود التكيف بذكاء وفعالية")
        print("🚀 نظام بصيرة جاهز للتطبيقات الحقيقية!")
        
    except Exception as e:
        print(f"❌ خطأ في العرض التوضيحي: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
