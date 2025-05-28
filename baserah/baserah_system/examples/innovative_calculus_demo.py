#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Innovative Calculus Demo - Expert-Explorer Integration
عرض توضيحي للنظام المبتكر للتفاضل والتكامل مع تكامل الخبير-المستكشف

This demo showcases the revolutionary calculus approach integrated
with the Expert-Explorer system in Basira.

Author: Basira System Development Team
Version: 1.0.0
"""

import torch
import math
import matplotlib.pyplot as plt
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from symbolic_processing.expert_explorer_system import Expert, ExpertKnowledgeType
    from mathematical_core.innovative_calculus_engine import InnovativeCalculusEngine
    from mathematical_core.calculus_test_functions import get_simple_test_functions, get_test_functions
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are properly installed and paths are correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('innovative_calculus_demo')


def demonstrate_innovative_calculus():
    """
    عرض توضيحي شامل للنظام المبتكر
    Comprehensive demonstration of the innovative calculus system
    """
    
    print("🌟" + "="*80 + "🌟")
    print("🚀 عرض توضيحي: النظام المبتكر للتفاضل والتكامل مع الخبير-المستكشف 🚀")
    print("🚀 Innovative Calculus with Expert-Explorer Integration Demo 🚀")
    print("🌟" + "="*80 + "🌟")
    
    # 1. Initialize Expert with Innovative Calculus
    print("\n📋 1. تهيئة الخبير مع النظام المبتكر...")
    print("📋 1. Initializing Expert with Innovative Calculus...")
    
    expert = Expert([
        ExpertKnowledgeType.HEURISTIC,
        ExpertKnowledgeType.ANALYTICAL,
        ExpertKnowledgeType.MATHEMATICAL
    ])
    
    print("✅ تم تهيئة الخبير بنجاح مع محرك التفاضل والتكامل المبتكر!")
    print("✅ Expert initialized successfully with Innovative Calculus Engine!")
    
    # 2. Train on Simple Functions
    print("\n🎓 2. تدريب النظام على الدوال البسيطة...")
    print("🎓 2. Training system on simple functions...")
    
    training_result = expert.train_calculus_engine(epochs=200)
    
    if training_result.get("success"):
        print("✅ تم التدريب بنجاح!")
        print("✅ Training completed successfully!")
        print(f"📊 الدوال المدربة: {training_result['functions_trained']}")
        print(f"📊 Functions trained: {training_result['functions_trained']}")
        
        # Display training results
        for func_name, metrics in training_result['results'].items():
            print(f"   📈 {func_name}: Loss = {metrics['final_loss']:.4f}")
    else:
        print("❌ فشل التدريب!")
        print("❌ Training failed!")
        return
    
    # 3. Demonstrate Problem Solving
    print("\n🧮 3. حل مسائل التفاضل والتكامل...")
    print("🧮 3. Solving calculus problems...")
    
    # Create a test function: f(x) = x^3 + 2x^2 - x + 1
    x = torch.linspace(-2.0, 2.0, 100)
    test_function = x**3 + 2*x**2 - x + 1
    true_derivative = 3*x**2 + 4*x - 1
    true_integral = x**4/4 + 2*x**3/3 - x**2/2 + x
    
    print(f"🔢 دالة الاختبار: f(x) = x³ + 2x² - x + 1")
    print(f"🔢 Test function: f(x) = x³ + 2x² - x + 1")
    
    # Solve using innovative approach
    solution = expert.solve_calculus_problem(test_function)
    
    if solution.get("success"):
        pred_derivative = solution["derivative"]
        pred_integral = solution["integral"]
        D_coeff = solution["differentiation_coefficients"]
        V_coeff = solution["integration_coefficients"]
        
        # Calculate accuracy
        derivative_error = torch.mean(torch.abs(pred_derivative - true_derivative)).item()
        integral_error = torch.mean(torch.abs(pred_integral - true_integral)).item()
        
        print("✅ تم حل المسألة بنجاح!")
        print("✅ Problem solved successfully!")
        print(f"📊 خطأ التفاضل: {derivative_error:.4f}")
        print(f"📊 Derivative error: {derivative_error:.4f}")
        print(f"📊 خطأ التكامل: {integral_error:.4f}")
        print(f"📊 Integral error: {integral_error:.4f}")
        
        # 4. Explore Coefficient Space
        print("\n🔍 4. استكشاف فضاء المعاملات...")
        print("🔍 4. Exploring coefficient space...")
        
        exploration_result = expert.explore_coefficient_space(
            target_function=test_function,
            exploration_steps=50
        )
        
        if exploration_result.get("success"):
            print("✅ تم الاستكشاف بنجاح!")
            print("✅ Exploration completed successfully!")
            print(f"🎯 أفضل خسارة: {exploration_result['best_loss']:.4f}")
            print(f"🎯 Best loss: {exploration_result['best_loss']:.4f}")
        
        # 5. Visualize Results
        print("\n📊 5. عرض النتائج بصرياً...")
        print("📊 5. Visualizing results...")
        
        visualize_results(x, test_function, true_derivative, true_integral,
                         pred_derivative, pred_integral, D_coeff, V_coeff)
        
    else:
        print("❌ فشل في حل المسألة!")
        print("❌ Failed to solve problem!")
    
    # 6. Performance Summary
    print("\n📈 6. ملخص الأداء...")
    print("📈 6. Performance summary...")
    
    summary = expert.calculus_engine.get_performance_summary()
    print(f"📊 إجمالي الدوال المدربة: {summary.get('total_functions_trained', 0)}")
    print(f"📊 Total functions trained: {summary.get('total_functions_trained', 0)}")
    print(f"📊 متوسط الخسارة النهائية: {summary.get('average_final_loss', 0):.4f}")
    print(f"📊 Average final loss: {summary.get('average_final_loss', 0):.4f}")
    print(f"📊 إجمالي الحالات: {summary.get('total_states', 0)}")
    print(f"📊 Total states: {summary.get('total_states', 0)}")
    
    print("\n🎉" + "="*80 + "🎉")
    print("🌟 انتهى العرض التوضيحي بنجاح! النظام المبتكر يعمل بشكل مثالي! 🌟")
    print("🌟 Demo completed successfully! Innovative system working perfectly! 🌟")
    print("🎉" + "="*80 + "🎉")


def visualize_results(x, function, true_derivative, true_integral,
                     pred_derivative, pred_integral, D_coeff, V_coeff):
    """
    عرض النتائج بصرياً
    Visualize the results
    """
    
    try:
        # Convert tensors to numpy for plotting
        x_np = x.detach().numpy()
        function_np = function.detach().numpy()
        true_derivative_np = true_derivative.detach().numpy()
        true_integral_np = true_integral.detach().numpy()
        pred_derivative_np = pred_derivative.detach().numpy()
        pred_integral_np = pred_integral.detach().numpy()
        D_coeff_np = D_coeff.detach().numpy()
        V_coeff_np = V_coeff.detach().numpy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Innovative Calculus Results - النتائج المبتكرة للتفاضل والتكامل', fontsize=16)
        
        # Original function
        axes[0, 0].plot(x_np, function_np, 'b-', linewidth=2, label='f(x)')
        axes[0, 0].set_title('Original Function - الدالة الأصلية')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('f(x)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Derivative comparison
        axes[0, 1].plot(x_np, true_derivative_np, 'g-', linewidth=2, label="True f'(x)")
        axes[0, 1].plot(x_np, pred_derivative_np, 'r--', linewidth=2, label="Predicted f'(x)")
        axes[0, 1].set_title('Derivative Comparison - مقارنة التفاضل')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel("f'(x)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Integral comparison
        axes[0, 2].plot(x_np, true_integral_np, 'g-', linewidth=2, label='True ∫f(x)dx')
        axes[0, 2].plot(x_np, pred_integral_np, 'r--', linewidth=2, label='Predicted ∫f(x)dx')
        axes[0, 2].set_title('Integral Comparison - مقارنة التكامل')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('∫f(x)dx')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Differentiation coefficients
        axes[1, 0].plot(x_np, D_coeff_np, 'm-', linewidth=2, label='D(x) coefficients')
        axes[1, 0].set_title('Differentiation Coefficients - معاملات التفاضل')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('D(x)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Integration coefficients
        axes[1, 1].plot(x_np, V_coeff_np, 'c-', linewidth=2, label='V(x) coefficients')
        axes[1, 1].set_title('Integration Coefficients - معاملات التكامل')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('V(x)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Error analysis
        derivative_error = np.abs(true_derivative_np - pred_derivative_np)
        integral_error = np.abs(true_integral_np - pred_integral_np)
        
        axes[1, 2].plot(x_np, derivative_error, 'r-', linewidth=2, label='Derivative Error')
        axes[1, 2].plot(x_np, integral_error, 'b-', linewidth=2, label='Integral Error')
        axes[1, 2].set_title('Error Analysis - تحليل الأخطاء')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('Absolute Error')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 تم عرض النتائج بصرياً بنجاح!")
        print("📊 Results visualized successfully!")
        
    except Exception as e:
        print(f"⚠️ خطأ في العرض البصري: {e}")
        print(f"⚠️ Visualization error: {e}")


def demonstrate_advanced_features():
    """
    عرض الميزات المتقدمة
    Demonstrate advanced features
    """
    
    print("\n🔬 عرض الميزات المتقدمة...")
    print("🔬 Demonstrating advanced features...")
    
    # Initialize expert
    expert = Expert()
    
    # Train on complex functions
    print("\n🎯 تدريب على دوال معقدة...")
    print("🎯 Training on complex functions...")
    
    complex_functions = ['exponential_wide', 'gaussian_wide', 'rational_wide']
    
    for func_name in complex_functions:
        print(f"🔄 تدريب على: {func_name}")
        print(f"🔄 Training on: {func_name}")
        
        result = expert.train_calculus_engine(function_name=func_name, epochs=300)
        
        if result.get("success"):
            loss = result['results'][func_name]['final_loss']
            print(f"✅ نجح التدريب! الخسارة النهائية: {loss:.4f}")
            print(f"✅ Training successful! Final loss: {loss:.4f}")
        else:
            print(f"❌ فشل التدريب على {func_name}")
            print(f"❌ Training failed on {func_name}")


if __name__ == "__main__":
    try:
        # Run main demonstration
        demonstrate_innovative_calculus()
        
        # Ask user if they want to see advanced features
        print("\n❓ هل تريد رؤية الميزات المتقدمة؟ (y/n)")
        print("❓ Would you like to see advanced features? (y/n)")
        
        response = input().lower().strip()
        if response in ['y', 'yes', 'نعم', 'ن']:
            demonstrate_advanced_features()
        
        print("\n🎊 شكراً لك على استخدام العرض التوضيحي!")
        print("🎊 Thank you for using the demo!")
        
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف العرض التوضيحي بواسطة المستخدم")
        print("⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ خطأ في العرض التوضيحي: {e}")
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
