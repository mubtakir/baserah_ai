#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Decomposition Engine - Revolutionary Series Expansion
محرك تفكيك الدوال - التوسع المتسلسل الثوري

This module implements the revolutionary function decomposition approach
discovered by Basil Yahya Abdullah, where any function can be decomposed
using the innovative series expansion method.

Original concept by: باسل يحيى عبدالله/ العراق/ الموصل
Integrated into Basira System by: Basira Development Team
Version: 1.0.0 (Revolutionary Integration)
"""

import numpy as np
import math
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core components
try:
    from core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
    from mathematical_core.innovative_calculus_engine import InnovativeCalculusEngine
except ImportError as e:
    logging.warning(f"Could not import required components: {e}")
    # Define placeholder classes
    class EquationType:
        DECOMPOSITION = "decomposition"
        SERIES = "series"
        SHAPE = "shape"
        PATTERN = "pattern"
        BEHAVIOR = "behavior"
        TRANSFORMATION = "transformation"

    class LearningMode:
        ADAPTIVE = "adaptive"
        REVOLUTIONARY = "revolutionary"
        NONE = "none"

    class GeneralShapeEquation:
        def __init__(self, equation_type, learning_mode):
            self.equation_type = equation_type
            self.learning_mode = learning_mode

# Configure logging
logger = logging.getLogger('mathematical_core.function_decomposition_engine')


@dataclass
class DecompositionState:
    """
    حالة تفكيك الدالة - تحتوي على معاملات المتسلسلة الثورية
    Function decomposition state containing revolutionary series coefficients
    """
    function_values: np.ndarray      # قيم الدالة الأصلية
    derivatives: List[np.ndarray]    # قائمة المشتقات
    series_coefficients: np.ndarray  # معاملات المتسلسلة
    integral_terms: List[np.ndarray] # الحدود التكاملية
    convergence_radius: float        # نصف قطر التقارب
    accuracy: float                  # دقة التقريب
    n_terms: int = 10                # عدد الحدود المستخدمة

    def evaluate_series(self, x: np.ndarray) -> np.ndarray:
        """
        تقييم المتسلسلة الثورية عند نقطة معينة
        Evaluate revolutionary series at given point
        """
        result = np.zeros_like(x)

        for n in range(1, self.n_terms + 1):
            if n-1 < len(self.derivatives):
                # الحد الأساسي: (-1)^(n-1) * (x^n * d^n A) / n!
                term = ((-1) ** (n-1)) * (x ** n) * self.derivatives[n-1] / math.factorial(n)
                result += term

                # الحد التكاملي: (-1)^n * ∫(x^n * d^(n+1) A) / n!
                if n < len(self.integral_terms):
                    integral_term = ((-1) ** n) * self.integral_terms[n-1] / math.factorial(n)
                    result += integral_term

        return result


class RevolutionarySeriesExpander:
    """
    موسع المتسلسلة الثورية - تطبيق فكرة باسل يحيى عبدالله
    Revolutionary Series Expander - Implementation of Basil Yahya Abdullah's idea
    """

    def __init__(self, max_terms: int = 20, tolerance: float = 1e-6):
        """
        Initialize the revolutionary series expander

        Args:
            max_terms: الحد الأقصى لعدد حدود المتسلسلة
            tolerance: التسامح في دقة التقارب
        """
        self.max_terms = max_terms
        self.tolerance = tolerance
        self.logger = logging.getLogger('mathematical_core.function_decomposition_engine.expander')

        # Initialize General Shape Equation for decomposition
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.DECOMPOSITION,
            learning_mode=LearningMode.REVOLUTIONARY
        )

        self.logger.info("Revolutionary Series Expander initialized")

    def compute_derivatives(self, function_values: np.ndarray, x: np.ndarray,
                          max_order: int) -> List[np.ndarray]:
        """
        حساب المشتقات العددية للدالة
        Compute numerical derivatives of the function
        """
        derivatives = []
        current_values = function_values.copy()

        for order in range(max_order):
            if order == 0:
                derivatives.append(current_values)
            else:
                # حساب المشتقة العددية
                h = (x[1] - x[0]) if len(x) > 1 else 1e-5
                derivative = np.zeros_like(current_values)

                # استخدام الفروق المحدودة المركزية
                for i in range(1, len(current_values) - 1):
                    derivative[i] = (current_values[i+1] - current_values[i-1]) / (2 * h)

                # معالجة الحدود
                derivative[0] = (current_values[1] - current_values[0]) / h
                derivative[-1] = (current_values[-1] - current_values[-2]) / h

                derivatives.append(derivative)
                current_values = derivative

        return derivatives

    def compute_integral_terms(self, derivatives: List[np.ndarray],
                             x: np.ndarray) -> List[np.ndarray]:
        """
        حساب الحدود التكاملية للمتسلسلة الثورية
        Compute integral terms for revolutionary series
        """
        integral_terms = []

        for n in range(len(derivatives) - 1):
            if n + 1 < len(derivatives):
                # حساب ∫(x^n * d^(n+1) A)
                integrand = (x ** (n + 1)) * derivatives[n + 1]

                # تكامل عددي باستخدام قاعدة شبه المنحرف
                h = (x[1] - x[0]) if len(x) > 1 else 1e-5
                integral = np.zeros_like(integrand)

                for i in range(1, len(integrand)):
                    integral[i] = integral[i-1] + h * (integrand[i] + integrand[i-1]) / 2

                integral_terms.append(integral)

        return integral_terms

    def decompose_function(self, function_values: np.ndarray,
                          x: np.ndarray) -> DecompositionState:
        """
        تفكيك الدالة باستخدام المتسلسلة الثورية
        Decompose function using revolutionary series

        Args:
            function_values: قيم الدالة
            x: قيم المتغير المستقل

        Returns:
            حالة التفكيك مع جميع المعاملات
        """
        self.logger.info("Starting revolutionary function decomposition")

        # حساب المشتقات
        derivatives = self.compute_derivatives(function_values, x, self.max_terms)

        # حساب الحدود التكاملية
        integral_terms = self.compute_integral_terms(derivatives, x)

        # حساب معاملات المتسلسلة
        series_coefficients = np.zeros(self.max_terms)
        for n in range(min(self.max_terms, len(derivatives))):
            series_coefficients[n] = np.mean(derivatives[n])

        # تقدير نصف قطر التقارب
        convergence_radius = self._estimate_convergence_radius(derivatives)

        # حساب الدقة
        reconstructed = self._reconstruct_function(derivatives, integral_terms, x)
        accuracy = self._calculate_accuracy(function_values, reconstructed)

        decomposition_state = DecompositionState(
            function_values=function_values,
            derivatives=derivatives,
            series_coefficients=series_coefficients,
            integral_terms=integral_terms,
            convergence_radius=convergence_radius,
            accuracy=accuracy,
            n_terms=min(self.max_terms, len(derivatives))
        )

        self.logger.info(f"Decomposition completed with accuracy: {accuracy:.6f}")
        return decomposition_state

    def _estimate_convergence_radius(self, derivatives: List[np.ndarray]) -> float:
        """تقدير نصف قطر التقارب"""
        if len(derivatives) < 2:
            return float('inf')

        # استخدام اختبار النسبة
        ratios = []
        for i in range(1, len(derivatives) - 1):
            ratio = np.mean(np.abs(derivatives[i+1])) / (np.mean(np.abs(derivatives[i])) + 1e-10)
            ratios.append(float(ratio))

        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            return 1.0 / avg_ratio if avg_ratio > 0 else float('inf')

        return float('inf')

    def _reconstruct_function(self, derivatives: List[np.ndarray],
                            integral_terms: List[np.ndarray],
                            x: np.ndarray) -> np.ndarray:
        """إعادة بناء الدالة من المتسلسلة"""
        result = np.zeros_like(x)

        for n in range(1, min(len(derivatives), self.max_terms) + 1):
            if n-1 < len(derivatives):
                # الحد الأساسي
                term = ((-1) ** (n-1)) * (x ** n) * derivatives[n-1] / math.factorial(n)
                result += term

                # الحد التكاملي
                if n-1 < len(integral_terms):
                    integral_term = ((-1) ** n) * integral_terms[n-1] / math.factorial(n)
                    result += integral_term

        return result

    def _calculate_accuracy(self, original: np.ndarray,
                          reconstructed: np.ndarray) -> float:
        """حساب دقة إعادة البناء"""
        mse = np.mean((original - reconstructed) ** 2)
        return 1.0 / (1.0 + float(mse))


class FunctionDecompositionEngine:
    """
    محرك تفكيك الدوال الرئيسي
    Main Function Decomposition Engine integrating with Basira System
    """

    def __init__(self, max_terms: int = 20, tolerance: float = 1e-6):
        """Initialize the function decomposition engine"""
        self.logger = logging.getLogger('mathematical_core.function_decomposition_engine.main')

        # Initialize General Shape Equation
        self.general_equation = GeneralShapeEquation(
            equation_type=EquationType.SERIES,
            learning_mode=LearningMode.ADAPTIVE
        )

        # Initialize the revolutionary series expander
        self.series_expander = RevolutionarySeriesExpander(max_terms, tolerance)

        # Integration with innovative calculus engine
        try:
            self.calculus_engine = InnovativeCalculusEngine()
        except:
            self.calculus_engine = None
            self.logger.warning("Could not initialize calculus engine integration")

        # Performance tracking
        self.decomposition_history = []
        self.performance_metrics = {}

        self.logger.info("Function Decomposition Engine initialized successfully")

    def decompose_function(self, function_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تفكيك دالة باستخدام المتسلسلة الثورية
        Decompose function using revolutionary series

        Args:
            function_data: Dictionary containing function information

        Returns:
            Decomposition results and analysis
        """
        self.logger.info(f"Decomposing function: {function_data.get('name', 'unnamed')}")

        try:
            # Extract function data
            if 'function' in function_data:
                f = function_data['function']
                domain = function_data.get('domain', (-2.0, 2.0, 100))
                start, end, num_points = domain

                x = np.linspace(start, end, num_points)
                function_values = f(x)
            else:
                x = function_data['x']
                function_values = function_data['values']

            # Perform decomposition
            decomposition_state = self.series_expander.decompose_function(function_values, x)

            # Analyze results
            analysis = self._analyze_decomposition(decomposition_state, x)

            # Store in history
            self.decomposition_history.append({
                'function_name': function_data.get('name', 'unnamed'),
                'timestamp': datetime.now().isoformat(),
                'accuracy': decomposition_state.accuracy,
                'n_terms': decomposition_state.n_terms,
                'convergence_radius': decomposition_state.convergence_radius
            })

            return {
                'success': True,
                'decomposition_state': decomposition_state,
                'analysis': analysis,
                'revolutionary_series': self._format_series_expression(decomposition_state),
                'performance': {
                    'accuracy': decomposition_state.accuracy,
                    'convergence_radius': decomposition_state.convergence_radius,
                    'n_terms_used': decomposition_state.n_terms
                }
            }

        except Exception as e:
            self.logger.error(f"Error in function decomposition: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_decomposition(self, state: DecompositionState,
                             x: np.ndarray) -> Dict[str, Any]:
        """تحليل نتائج التفكيك"""
        analysis = {
            'convergence_quality': 'excellent' if state.convergence_radius > 10 else 'good' if state.convergence_radius > 1 else 'limited',
            'accuracy_level': 'high' if state.accuracy > 0.95 else 'medium' if state.accuracy > 0.8 else 'low',
            'series_efficiency': state.n_terms / self.series_expander.max_terms,
            'derivative_analysis': self._analyze_derivatives(state.derivatives),
            'integral_analysis': self._analyze_integrals(state.integral_terms)
        }

        return analysis

    def _analyze_derivatives(self, derivatives: List[np.ndarray]) -> Dict[str, Any]:
        """تحليل المشتقات"""
        if not derivatives:
            return {'status': 'no_derivatives'}

        # حساب معدلات التغير
        decay_rates = []
        for i in range(1, len(derivatives)):
            ratio = np.mean(np.abs(derivatives[i])) / (np.mean(np.abs(derivatives[i-1])) + 1e-10)
            decay_rates.append(float(ratio))

        return {
            'num_derivatives': len(derivatives),
            'decay_pattern': 'exponential' if all(r < 0.5 for r in decay_rates[-3:]) else 'polynomial',
            'average_decay_rate': sum(decay_rates) / len(decay_rates) if decay_rates else 0,
            'smoothness_indicator': len([r for r in decay_rates if r < 0.1])
        }

    def _analyze_integrals(self, integral_terms: List[np.ndarray]) -> Dict[str, Any]:
        """تحليل الحدود التكاملية"""
        if not integral_terms:
            return {'status': 'no_integrals'}

        # حساب مساهمة كل حد تكاملي
        contributions = []
        for term in integral_terms:
            contribution = float(np.mean(np.abs(term)))
            contributions.append(contribution)

        return {
            'num_integral_terms': len(integral_terms),
            'total_contribution': sum(contributions),
            'dominant_terms': len([c for c in contributions if c > 0.1 * max(contributions)]),
            'contribution_distribution': contributions[:5]  # أول 5 حدود
        }

    def _format_series_expression(self, state: DecompositionState) -> str:
        """تنسيق تعبير المتسلسلة"""
        expression = "A(x) = "

        for n in range(1, min(state.n_terms + 1, 6)):  # عرض أول 5 حدود
            sign = "+" if (n-1) % 2 == 0 else "-"
            if n == 1:
                sign = "" if sign == "+" else "-"

            expression += f"{sign} (x^{n} * d^{n}A) / {n}!"

            if n < min(state.n_terms, 5):
                expression += " "

        if state.n_terms > 5:
            expression += " + ..."

        return expression

    def get_performance_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص الأداء"""
        if not self.decomposition_history:
            return {"message": "No decomposition history available"}

        accuracies = [h['accuracy'] for h in self.decomposition_history]
        convergence_radii = [h['convergence_radius'] for h in self.decomposition_history if h['convergence_radius'] != float('inf')]

        summary = {
            "total_decompositions": len(self.decomposition_history),
            "average_accuracy": sum(accuracies) / len(accuracies),
            "best_accuracy": max(accuracies),
            "average_convergence_radius": sum(convergence_radii) / len(convergence_radii) if convergence_radii else 0,
            "recent_decompositions": self.decomposition_history[-5:]
        }

        return summary
