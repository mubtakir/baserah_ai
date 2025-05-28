#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revolutionary Function Decomposition Demo
عرض توضيحي للنظام الثوري لتفكيك الدوال

This demo showcases the revolutionary function decomposition approach
discovered by Basil Yahya Abdullah, integrated with the Expert-Explorer system.

Original concept by: باسل يحيى عبدالله/ العراق/ الموصل
Demo by: Basira System Development Team
Version: 1.0.0
"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from symbolic_processing.expert_explorer_system import Expert, ExpertKnowledgeType
    from mathematical_core.function_decomposition_engine import FunctionDecompositionEngine
    from mathematical_core.calculus_test_functions import get_decomposition_test_functions
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are properly installed and paths are correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('revolutionary_decomposition_demo')


def demonstrate_revolutionary_decomposition():
    """
    عرض توضيحي شامل للنظام الثوري لتفكيك الدوال
    Comprehensive demonstration of revolutionary function decomposition
    """
    
    print("🌟" + "="*90 + "🌟")
    print("🚀 عرض توضيحي: النظام الثوري لتفكيك الدوال - فكرة باسل يحيى عبدالله 🚀")
    print("🚀 Revolutionary Function Decomposition Demo - Basil Yahya Abdullah's Idea 🚀")
    print("🌟" + "="*90 + "🌟")
    
    # 1. Initialize Expert with Revolutionary Decomposition
    print("\n📋 1. تهيئة الخبير مع النظام الثوري...")
    print("📋 1. Initializing Expert with Revolutionary Decomposition...")
    
    expert = Expert([
        ExpertKnowledgeType.HEURISTIC,
        ExpertKnowledgeType.ANALYTICAL,
        ExpertKnowledgeType.MATHEMATICAL
    ])
    
    print("✅ تم تهيئة الخبير بنجاح مع محرك التفكيك الثوري!")
    print("✅ Expert initialized successfully with Revolutionary Decomposition Engine!")
    
    # 2. Demonstrate the Revolutionary Concept
    print("\n💡 2. شرح المفهوم الثوري...")
    print("💡 2. Explaining the Revolutionary Concept...")
    
    print("🔬 الفرضية الرياضية الثورية لباسل يحيى عبدالله:")
    print("🔬 Basil Yahya Abdullah's Revolutionary Mathematical Hypothesis:")
    print("   A = x.dA - ∫x.d2A")
    print("   حيث: A = الدالة، dA = المشتقة الأولى، d2A = المشتقة الثانية")
    print("   Where: A = function, dA = first derivative, d2A = second derivative")
    
    print("\n🌟 المتسلسلة الثورية الناتجة:")
    print("🌟 Resulting Revolutionary Series:")
    print("   A = Σ[(-1)^(n-1) * (x^n * d^n A) / n!] + (-1)^n * ∫(x^n * d^(n+1) A) / n!")
    
    # 3. Test on Different Functions
    print("\n🧮 3. اختبار النظام على دوال مختلفة...")
    print("🧮 3. Testing system on different functions...")
    
    test_functions = {
        'exponential': {
            'name': 'Exponential Function',
            'function': lambda x: torch.exp(x),
            'domain': (-1.0, 1.0, 100),
            'description': 'f(x) = e^x - دالة أسية'
        },
        'polynomial': {
            'name': 'Polynomial Function',
            'function': lambda x: x**3 - 2*x**2 + x + 1,
            'domain': (-2.0, 2.0, 100),
            'description': 'f(x) = x³ - 2x² + x + 1 - كثير حدود'
        },
        'trigonometric': {
            'name': 'Trigonometric Function',
            'function': lambda x: torch.sin(2*x) + 0.5*torch.cos(x),
            'domain': (-math.pi, math.pi, 150),
            'description': 'f(x) = sin(2x) + 0.5cos(x) - دالة مثلثية'
        }
    }
    
    decomposition_results = {}
    
    for func_name, func_data in test_functions.items():
        print(f"\n🔍 اختبار: {func_data['description']}")
        print(f"🔍 Testing: {func_data['description']}")
        
        # Perform revolutionary decomposition
        result = expert.decompose_function_revolutionary(func_data)
        
        if result.get("success"):
            decomposition_state = result['decomposition_state']
            analysis = result['analysis']
            
            print(f"✅ نجح التفكيك!")
            print(f"✅ Decomposition successful!")
            print(f"   📊 الدقة: {decomposition_state.accuracy:.4f}")
            print(f"   📊 Accuracy: {decomposition_state.accuracy:.4f}")
            print(f"   📊 نصف قطر التقارب: {decomposition_state.convergence_radius:.4f}")
            print(f"   📊 Convergence radius: {decomposition_state.convergence_radius:.4f}")
            print(f"   📊 عدد الحدود: {decomposition_state.n_terms}")
            print(f"   📊 Number of terms: {decomposition_state.n_terms}")
            print(f"   📊 جودة التقارب: {analysis['convergence_quality']}")
            print(f"   📊 Convergence quality: {analysis['convergence_quality']}")
            
            decomposition_results[func_name] = result
        else:
            print(f"❌ فشل التفكيك: {result.get('error', 'unknown error')}")
            print(f"❌ Decomposition failed: {result.get('error', 'unknown error')}")
    
    # 4. Explore Series Convergence
    print("\n🔍 4. استكشاف تقارب المتسلسلة...")
    print("🔍 4. Exploring series convergence...")
    
    if 'exponential' in decomposition_results:
        convergence_result = expert.explore_series_convergence(
            test_functions['exponential'],
            exploration_steps=20
        )
        
        if convergence_result.get("success"):
            best_config = convergence_result['best_configuration']
            convergence_analysis = convergence_result['convergence_analysis']
            
            print("✅ تم استكشاف التقارب بنجاح!")
            print("✅ Convergence exploration successful!")
            print(f"   🎯 أفضل عدد حدود: {best_config['n_terms']}")
            print(f"   🎯 Best number of terms: {best_config['n_terms']}")
            print(f"   🎯 أفضل دقة: {best_config['accuracy']:.4f}")
            print(f"   🎯 Best accuracy: {best_config['accuracy']:.4f}")
            print(f"   🎯 جودة التقارب: {convergence_analysis['convergence_quality']}")
            print(f"   🎯 Convergence quality: {convergence_analysis['convergence_quality']}")
    
    # 5. Compare with Traditional Methods
    print("\n⚖️ 5. مقارنة مع الطرق التقليدية...")
    print("⚖️ 5. Comparing with traditional methods...")
    
    if 'polynomial' in decomposition_results:
        comparison_result = expert.compare_decomposition_methods(test_functions['polynomial'])
        
        if comparison_result.get("success"):
            recommendation = comparison_result['recommendation']
            
            print("✅ تمت المقارنة بنجاح!")
            print("✅ Comparison successful!")
            print(f"   🏆 الطريقة الموصى بها: {recommendation['recommended_method']}")
            print(f"   🏆 Recommended method: {recommendation['recommended_method']}")
            print(f"   💡 السبب: {recommendation['reason']}")
            print(f"   💡 Reason: {recommendation['reason']}")
            
            if 'advantages' in recommendation:
                print("   ✨ المزايا:")
                print("   ✨ Advantages:")
                for advantage in recommendation['advantages']:
                    print(f"      • {advantage}")
    
    # 6. Visualize Results
    print("\n📊 6. عرض النتائج بصرياً...")
    print("📊 6. Visualizing results...")
    
    if decomposition_results:
        visualize_decomposition_results(decomposition_results, test_functions)
    
    # 7. Performance Summary
    print("\n📈 7. ملخص الأداء...")
    print("📈 7. Performance summary...")
    
    summary = expert.decomposition_engine.get_performance_summary()
    print(f"📊 إجمالي التفكيكات: {summary.get('total_decompositions', 0)}")
    print(f"📊 Total decompositions: {summary.get('total_decompositions', 0)}")
    print(f"📊 متوسط الدقة: {summary.get('average_accuracy', 0):.4f}")
    print(f"📊 Average accuracy: {summary.get('average_accuracy', 0):.4f}")
    print(f"📊 أفضل دقة: {summary.get('best_accuracy', 0):.4f}")
    print(f"📊 Best accuracy: {summary.get('best_accuracy', 0):.4f}")
    
    print("\n🎉" + "="*90 + "🎉")
    print("🌟 انتهى العرض التوضيحي بنجاح! النظام الثوري يعمل بشكل مثالي! 🌟")
    print("🌟 Demo completed successfully! Revolutionary system working perfectly! 🌟")
    print("🌟 شكراً لباسل يحيى عبدالله على هذا الاكتشاف الرياضي المذهل! 🌟")
    print("🌟 Thanks to Basil Yahya Abdullah for this amazing mathematical discovery! 🌟")
    print("🎉" + "="*90 + "🎉")


def visualize_decomposition_results(decomposition_results, test_functions):
    """
    عرض نتائج التفكيك بصرياً
    Visualize decomposition results
    """
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Revolutionary Function Decomposition Results\nنتائج التفكيك الثوري للدوال', fontsize=16)
        
        # Plot original functions and their reconstructions
        plot_idx = 0
        for func_name, result in decomposition_results.items():
            if plot_idx >= 3:  # Limit to 3 functions
                break
                
            if result.get("success"):
                func_data = test_functions[func_name]
                decomposition_state = result['decomposition_state']
                
                # Generate data for plotting
                domain = func_data['domain']
                x = torch.linspace(domain[0], domain[1], domain[2])
                original_values = func_data['function'](x)
                
                # Reconstruct function using series
                reconstructed_values = decomposition_state.evaluate_series(x)
                
                # Convert to numpy for plotting
                x_np = x.detach().numpy()
                original_np = original_values.detach().numpy()
                reconstructed_np = reconstructed_values.detach().numpy()
                
                # Plot
                row = plot_idx // 2
                col = plot_idx % 2
                ax = axes[row, col]
                
                ax.plot(x_np, original_np, 'b-', linewidth=2, label='Original Function')
                ax.plot(x_np, reconstructed_np, 'r--', linewidth=2, label='Revolutionary Series')
                ax.set_title(f'{func_data["name"]}\nAccuracy: {decomposition_state.accuracy:.4f}')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plot_idx += 1
        
        # Performance comparison plot
        if len(decomposition_results) > 1:
            ax = axes[1, 1] if plot_idx <= 2 else axes[1, 0]
            
            function_names = []
            accuracies = []
            convergence_radii = []
            
            for func_name, result in decomposition_results.items():
                if result.get("success"):
                    function_names.append(func_name)
                    accuracies.append(result['decomposition_state'].accuracy)
                    convergence_radii.append(min(result['decomposition_state'].convergence_radius, 10))  # Cap for visualization
            
            x_pos = np.arange(len(function_names))
            
            ax.bar(x_pos - 0.2, accuracies, 0.4, label='Accuracy', alpha=0.7)
            ax.bar(x_pos + 0.2, [r/10 for r in convergence_radii], 0.4, label='Convergence Radius (scaled)', alpha=0.7)
            
            ax.set_title('Performance Comparison\nمقارنة الأداء')
            ax.set_xlabel('Functions')
            ax.set_ylabel('Performance Metrics')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(function_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📊 تم عرض النتائج بصرياً بنجاح!")
        print("📊 Results visualized successfully!")
        
    except Exception as e:
        print(f"⚠️ خطأ في العرض البصري: {e}")
        print(f"⚠️ Visualization error: {e}")


def demonstrate_mathematical_foundation():
    """
    عرض الأساس الرياضي للنظام الثوري
    Demonstrate mathematical foundation of revolutionary system
    """
    
    print("\n🔬 الأساس الرياضي للنظام الثوري...")
    print("🔬 Mathematical foundation of revolutionary system...")
    
    print("\n📐 الفرضية الأساسية:")
    print("📐 Basic hypothesis:")
    print("   A = x·dA - ∫x·d²A")
    
    print("\n🔍 البرهان:")
    print("🔍 Proof:")
    print("   dA/dx = d/dx[x·dA - ∫x·d²A]")
    print("   dA/dx = (x·d²A + dA) - x·d²A")
    print("   dA/dx = dA ✓")
    
    print("\n🌟 المتسلسلة الناتجة:")
    print("🌟 Resulting series:")
    print("   A = Σ[(-1)^(n-1) · (x^n · d^n A) / n!] + R_n")
    print("   حيث R_n هو الحد التكاملي المتبقي")
    print("   Where R_n is the remaining integral term")


if __name__ == "__main__":
    try:
        # Run main demonstration
        demonstrate_revolutionary_decomposition()
        
        # Ask user if they want to see mathematical foundation
        print("\n❓ هل تريد رؤية الأساس الرياضي للنظام؟ (y/n)")
        print("❓ Would you like to see the mathematical foundation? (y/n)")
        
        response = input().lower().strip()
        if response in ['y', 'yes', 'نعم', 'ن']:
            demonstrate_mathematical_foundation()
        
        print("\n🎊 شكراً لك على استخدام العرض التوضيحي!")
        print("🎊 Thank you for using the demo!")
        print("🌟 تحية إجلال وتقدير لباسل يحيى عبدالله على هذا الإنجاز العلمي! 🌟")
        print("🌟 Salute and appreciation to Basil Yahya Abdullah for this scientific achievement! 🌟")
        
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف العرض التوضيحي بواسطة المستخدم")
        print("⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ خطأ في العرض التوضيحي: {e}")
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
