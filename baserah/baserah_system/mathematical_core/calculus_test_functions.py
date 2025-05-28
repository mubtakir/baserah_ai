#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Functions for Innovative Calculus Engine
دوال الاختبار لمحرك التفاضل والتكامل المبتكر

This module contains comprehensive test functions for evaluating
the innovative calculus engine performance.

Author: Basira System Development Team
Version: 1.0.0
"""

import torch
import math
from typing import Dict, Any, Callable


def get_test_functions() -> Dict[str, Dict[str, Any]]:
    """
    إرجاع مجموعة شاملة من دوال الاختبار
    Return comprehensive set of test functions for calculus engine

    Returns:
        Dictionary of test functions with their derivatives, integrals, and domains
    """

    test_functions = {
        'exponential_wide': {
            'name': 'Exponential Function (Wide Range)',
            'f': lambda x: torch.exp(x),
            'f_prime': lambda x: torch.exp(x),
            'f_integral': lambda x: torch.exp(x) - 1,
            'domain': (-2.0, 4.0, 200),
            'noise': 0.2,
            'description': 'دالة أسية مع نطاق واسع'
        },

        'sine_wide': {
            'name': 'Sine Function (Wide Range)',
            'f': lambda x: torch.sin(x),
            'f_prime': lambda x: torch.cos(x),
            'f_integral': lambda x: 1 - torch.cos(x),
            'domain': (0.0, 2*math.pi, 200),
            'noise': 0.2,
            'description': 'دالة جيبية مع نطاق واسع'
        },

        'cubic_wide': {
            'name': 'Cubic Function (Wide Range)',
            'f': lambda x: x**3,
            'f_prime': lambda x: 3*x**2,
            'f_integral': lambda x: x**4 / 4,
            'domain': (-2.0, 2.0, 200),
            'noise': 0.2,
            'description': 'دالة تكعيبية مع نطاق واسع'
        },

        'cosine_wide': {
            'name': 'Cosine Function (Wide Range)',
            'f': lambda x: torch.cos(x),
            'f_prime': lambda x: -torch.sin(x),
            'f_integral': lambda x: torch.sin(x),
            'domain': (0.0, math.pi, 200),
            'noise': 0.2,
            'description': 'دالة جيب التمام مع نطاق واسع'
        },

        'piecewise_complex': {
            'name': 'Complex Piecewise Function',
            'f': lambda x: torch.where(x < 0, torch.sin(x), torch.where(x < 1, x**2, torch.cos(x))),
            'f_prime': lambda x: torch.where(x < 0, torch.cos(x), torch.where(x < 1, 2*x, -torch.sin(x))),
            'f_integral': lambda x: torch.where(
                x < 0,
                -torch.cos(x) + 1,
                torch.where(
                    x < 1,
                    x**3/3 + 1 - 1,
                    torch.sin(x) + (torch.tensor(1.0)/3.0) - torch.cos(torch.tensor(1.0))
                )
            ),
            'domain': (-1.0, 2.0, 200),
            'noise': 0.3,
            'description': 'دالة مقطعية معقدة'
        },

        'absolute_wide': {
            'name': 'Absolute Value Function (Wide Range)',
            'f': lambda x: torch.abs(x),
            'f_prime': lambda x: torch.sign(x),
            'f_integral': lambda x: 0.5 * x * torch.abs(x),
            'domain': (-4.0, 4.0, 200),
            'noise': 0.2,
            'description': 'دالة القيمة المطلقة مع نطاق واسع'
        },

        'sqrt_wide': {
            'name': 'Square Root Function (Wide Range)',
            'f': lambda x: torch.sqrt(x),
            'f_prime': lambda x: 0.5 * x**-0.5,
            'f_integral': lambda x: (2/3) * x**1.5,
            'domain': (0.01, 4.0, 200),
            'noise': 0.2,
            'description': 'دالة الجذر التربيعي مع نطاق واسع'
        },

        'gaussian_wide': {
            'name': 'Gaussian Function (Wide Range)',
            'f': lambda x: torch.exp(-x**2),
            'f_prime': lambda x: -2*x*torch.exp(-x**2),
            'f_integral': lambda x: torch.sqrt(torch.tensor(math.pi)) * 0.5 * torch.erf(x),
            'domain': (-4.0, 4.0, 200),
            'noise': 0.2,
            'description': 'دالة جاوس مع نطاق واسع'
        },

        'rational_wide': {
            'name': 'Rational Function (Wide Range)',
            'f': lambda x: x / (x**2 + 1),
            'f_prime': lambda x: (1 - x**2) / (x**2 + 1)**2,
            'f_integral': lambda x: 0.5 * torch.log(x**2 + 1),
            'domain': (-6.0, 6.0, 200),
            'noise': 0.2,
            'description': 'دالة كسرية مع نطاق واسع'
        },

        'combined_complex': {
            'name': 'Combined Complex Function',
            'f': lambda x: torch.sin(2*x) + 0.5 * x**3 * torch.exp(-0.5*x) + torch.abs(x-1),
            'f_prime': lambda x: (2*torch.cos(2*x) +
                                1.5 * x**2 * torch.exp(-0.5*x) -
                                0.25 * x**3 * torch.exp(-0.5*x) +
                                torch.sign(x-1)),
            'f_integral': lambda x: (-0.5*torch.cos(2*x) +
                                   (-x**3 - 6*x**2 - 24*x - 48) * torch.exp(-0.5*x) - 0.5 +
                                   0.5 * (x-1) * torch.abs(x-1)),
            'domain': (-2.0, 4.0, 200),
            'noise': 0.4,
            'description': 'دالة مركبة معقدة جداً'
        },

        'non_differentiable': {
            'name': 'Non-Differentiable Function',
            'f': lambda x: torch.where(x < 0, x**2, torch.sqrt(torch.abs(x) + 1e-8)),
            'f_prime': lambda x: torch.where(x < 0, 2*x, 0.5 * (torch.abs(x) + 1e-8)**-0.5),
            'f_integral': lambda x: torch.where(x < 0, x**3/3, (2/3) * (torch.abs(x) + 1e-8)**1.5),
            'domain': (-2.0, 2.0, 200),
            'noise': 0.3,
            'description': 'دالة غير قابلة للاشتقاق في نقطة'
        },

        'logarithmic': {
            'name': 'Logarithmic Function',
            'f': lambda x: torch.log(x + 1e-8),
            'f_prime': lambda x: 1 / (x + 1e-8),
            'f_integral': lambda x: (x + 1e-8) * torch.log(x + 1e-8) - x,
            'domain': (0.1, 5.0, 200),
            'noise': 0.2,
            'description': 'دالة لوغاريتمية'
        },

        'polynomial_high': {
            'name': 'High-Degree Polynomial',
            'f': lambda x: x**5 - 3*x**4 + 2*x**3 - x**2 + 4*x - 1,
            'f_prime': lambda x: 5*x**4 - 12*x**3 + 6*x**2 - 2*x + 4,
            'f_integral': lambda x: x**6/6 - 3*x**5/5 + x**4/2 - x**3/3 + 2*x**2 - x,
            'domain': (-2.0, 2.0, 200),
            'noise': 0.2,
            'description': 'كثير حدود من الدرجة الخامسة'
        },

        'hyperbolic': {
            'name': 'Hyperbolic Functions',
            'f': lambda x: torch.sinh(x),
            'f_prime': lambda x: torch.cosh(x),
            'f_integral': lambda x: torch.cosh(x) - 1,
            'domain': (-2.0, 2.0, 200),
            'noise': 0.2,
            'description': 'دالة الجيب الزائدي'
        },

        'oscillatory': {
            'name': 'Highly Oscillatory Function',
            'f': lambda x: torch.sin(10*x) * torch.exp(-x**2),
            'f_prime': lambda x: (10*torch.cos(10*x) - 2*x*torch.sin(10*x)) * torch.exp(-x**2),
            'f_integral': lambda x: -0.5 * torch.sqrt(torch.tensor(math.pi)) * torch.erf(x) * torch.sin(10*x),  # تقريبي
            'domain': (-2.0, 2.0, 400),
            'noise': 0.3,
            'description': 'دالة عالية التذبذب'
        }
    }

    return test_functions


def get_simple_test_functions() -> Dict[str, Dict[str, Any]]:
    """
    إرجاع مجموعة مبسطة من دوال الاختبار للاختبار السريع
    Return simplified set of test functions for quick testing
    """

    simple_functions = {
        'linear': {
            'name': 'Linear Function',
            'f': lambda x: 2*x + 1,
            'f_prime': lambda x: torch.full_like(x, 2.0),
            'f_integral': lambda x: x**2 + x,
            'domain': (-1.0, 1.0, 50),
            'noise': 0.1,
            'description': 'دالة خطية بسيطة'
        },

        'quadratic': {
            'name': 'Quadratic Function',
            'f': lambda x: x**2,
            'f_prime': lambda x: 2*x,
            'f_integral': lambda x: x**3 / 3,
            'domain': (-2.0, 2.0, 50),
            'noise': 0.1,
            'description': 'دالة تربيعية بسيطة'
        },

        'sine_simple': {
            'name': 'Simple Sine',
            'f': lambda x: torch.sin(x),
            'f_prime': lambda x: torch.cos(x),
            'f_integral': lambda x: 1 - torch.cos(x),
            'domain': (0.0, math.pi, 50),
            'noise': 0.1,
            'description': 'دالة جيبية بسيطة'
        }
    }

    return simple_functions


def calculate_mae(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """حساب متوسط الخطأ المطلق"""
    return torch.mean(torch.abs(predicted - true)).item()


def calculate_mse(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """حساب متوسط مربع الخطأ"""
    return torch.mean((predicted - true)**2).item()


def calculate_r_squared(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """حساب معامل التحديد R²"""
    ss_res = torch.sum((true - predicted) ** 2)
    ss_tot = torch.sum((true - torch.mean(true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared.item()


def evaluate_performance(predicted_derivative: torch.Tensor,
                        true_derivative: torch.Tensor,
                        predicted_integral: torch.Tensor,
                        true_integral: torch.Tensor) -> Dict[str, float]:
    """
    تقييم شامل لأداء المحرك
    Comprehensive performance evaluation
    """

    metrics = {
        # مقاييس التفاضل
        'derivative_mae': calculate_mae(predicted_derivative, true_derivative),
        'derivative_mse': calculate_mse(predicted_derivative, true_derivative),
        'derivative_r2': calculate_r_squared(predicted_derivative, true_derivative),

        # مقاييس التكامل
        'integral_mae': calculate_mae(predicted_integral, true_integral),
        'integral_mse': calculate_mse(predicted_integral, true_integral),
        'integral_r2': calculate_r_squared(predicted_integral, true_integral),

        # مقاييس إجمالية
        'total_mae': (calculate_mae(predicted_derivative, true_derivative) +
                     calculate_mae(predicted_integral, true_integral)) / 2,
        'total_mse': (calculate_mse(predicted_derivative, true_derivative) +
                     calculate_mse(predicted_integral, true_integral)) / 2
    }

    return metrics


def get_decomposition_test_functions() -> Dict[str, Dict[str, Any]]:
    """
    دوال اختبار خاصة بمحرك تفكيك الدوال الثوري
    Test functions specifically for revolutionary function decomposition engine

    Returns:
        Dictionary of test functions for decomposition testing
    """

    decomposition_functions = {
        'exponential_decomp': {
            'name': 'Exponential Function for Decomposition',
            'function': lambda x: torch.exp(x),
            'domain': (-1.0, 1.0, 100),
            'expected_convergence': 'infinite',
            'description': 'دالة أسية لاختبار التفكيك - تقارب لا نهائي'
        },

        'polynomial_decomp': {
            'name': 'Polynomial Function for Decomposition',
            'function': lambda x: x**4 - 2*x**3 + x**2 - x + 1,
            'domain': (-2.0, 2.0, 100),
            'expected_convergence': 'infinite',
            'description': 'كثير حدود لاختبار التفكيك - تقارب لا نهائي'
        },

        'trigonometric_decomp': {
            'name': 'Trigonometric Function for Decomposition',
            'function': lambda x: torch.sin(x) + 0.5 * torch.cos(2*x),
            'domain': (-math.pi, math.pi, 150),
            'expected_convergence': 'infinite',
            'description': 'دالة مثلثية مركبة لاختبار التفكيك'
        },

        'rational_decomp': {
            'name': 'Rational Function for Decomposition',
            'function': lambda x: 1 / (1 + x**2),
            'domain': (-0.8, 0.8, 100),
            'expected_convergence': 'limited',
            'description': 'دالة كسرية لاختبار التفكيك - تقارب محدود'
        },

        'logarithmic_decomp': {
            'name': 'Logarithmic Function for Decomposition',
            'function': lambda x: torch.log(1 + x),
            'domain': (0.1, 2.0, 100),
            'expected_convergence': 'limited',
            'description': 'دالة لوغاريتمية لاختبار التفكيك'
        },

        'hyperbolic_decomp': {
            'name': 'Hyperbolic Function for Decomposition',
            'function': lambda x: torch.sinh(x) + torch.cosh(x/2),
            'domain': (-1.5, 1.5, 120),
            'expected_convergence': 'good',
            'description': 'دالة زائدية مركبة لاختبار التفكيك'
        },

        'oscillatory_decomp': {
            'name': 'Oscillatory Function for Decomposition',
            'function': lambda x: torch.sin(3*x) * torch.exp(-x**2/4),
            'domain': (-3.0, 3.0, 200),
            'expected_convergence': 'excellent',
            'description': 'دالة متذبذبة مع تخميد لاختبار التفكيك'
        },

        'complex_decomp': {
            'name': 'Complex Mixed Function for Decomposition',
            'function': lambda x: torch.exp(-x**2/2) * torch.sin(2*x) + x**3 * torch.exp(-x),
            'domain': (-2.0, 2.0, 150),
            'expected_convergence': 'good',
            'description': 'دالة مركبة معقدة لاختبار التفكيك المتقدم'
        }
    }

    return decomposition_functions


def evaluate_decomposition_performance(original_function: torch.Tensor,
                                     reconstructed_function: torch.Tensor,
                                     decomposition_state) -> Dict[str, Any]:
    """
    تقييم أداء تفكيك الدالة
    Evaluate function decomposition performance

    Args:
        original_function: الدالة الأصلية
        reconstructed_function: الدالة المعاد بناؤها
        decomposition_state: حالة التفكيك

    Returns:
        مقاييس الأداء الشاملة
    """

    # حساب الأخطاء الأساسية
    mae = calculate_mae(reconstructed_function, original_function)
    mse = calculate_mse(reconstructed_function, original_function)
    r2 = calculate_r_squared(reconstructed_function, original_function)

    # حساب الأخطاء النسبية
    relative_error = torch.mean(torch.abs((original_function - reconstructed_function) /
                                        (original_function + 1e-10))).item()

    # تحليل التقارب
    convergence_analysis = {
        'radius': decomposition_state.convergence_radius,
        'quality': 'excellent' if decomposition_state.convergence_radius > 10 else
                  'good' if decomposition_state.convergence_radius > 1 else 'limited'
    }

    # تحليل الكفاءة
    efficiency_analysis = {
        'terms_used': decomposition_state.n_terms,
        'accuracy_per_term': decomposition_state.accuracy / decomposition_state.n_terms,
        'series_efficiency': decomposition_state.accuracy * (1 - decomposition_state.n_terms / 20)
    }

    # تحليل الاستقرار العددي
    stability_analysis = {
        'numerical_stability': 'stable' if mae < 0.1 else 'moderate' if mae < 1.0 else 'unstable',
        'reconstruction_fidelity': r2,
        'error_distribution': torch.std(torch.abs(original_function - reconstructed_function)).item()
    }

    return {
        'basic_metrics': {
            'mae': mae,
            'mse': mse,
            'r_squared': r2,
            'relative_error': relative_error
        },
        'convergence_analysis': convergence_analysis,
        'efficiency_analysis': efficiency_analysis,
        'stability_analysis': stability_analysis,
        'overall_score': (r2 + (1 - relative_error) + decomposition_state.accuracy) / 3
    }
