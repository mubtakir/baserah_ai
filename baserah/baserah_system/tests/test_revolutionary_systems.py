#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار شامل للأنظمة الثورية - مراقبة التعلم والأداء
Comprehensive Test for Revolutionary Systems - Learning and Performance Monitoring

هذا الاختبار يراقب:
- هل الأنظمة الثورية تتعلم؟
- هل الاستبدال حقق قفزة في الأداء؟
- هل المعادلات المتكيفة تتطور؟
- هل منهجية باسل تعمل بفعالية؟

Author: Basil Yahya Abdullah - Iraq/Mosul
"""

import sys
import os
import time
import numpy as np
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import revolutionary systems
try:
    from learning.reinforcement.innovative_rl import (
        RevolutionaryExpertExplorerSystem,
        RevolutionaryLearningConfig,
        RevolutionaryExperience,
        RevolutionaryRewardType,
        RevolutionaryLearningStrategy
    )
    from learning.reinforcement.equation_based_rl import (
        RevolutionaryAdaptiveEquationSystem,
        RevolutionaryAdaptiveConfig,
        RevolutionaryAdaptiveExperience
    )
    from learning.innovative_reinforcement.agent import (
        RevolutionaryExpertExplorerAgent,
        RevolutionaryAgentConfig,
        RevolutionaryDecisionStrategy
    )
    REVOLUTIONARY_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ خطأ في استيراد الأنظمة الثورية: {e}")
    REVOLUTIONARY_SYSTEMS_AVAILABLE = False

    # Define placeholder classes for testing
    class RevolutionaryLearningStrategy:
        BASIL_INTEGRATIVE = "basil_integrative"

    class RevolutionaryDecisionStrategy:
        BASIL_INTEGRATIVE = "basil_integrative"


class RevolutionarySystemTester:
    """فاحص الأنظمة الثورية - مراقب التعلم والأداء"""

    def __init__(self):
        """تهيئة الفاحص"""
        print("🌟" + "="*80 + "🌟")
        print("🔬 فاحص الأنظمة الثورية - مراقب التعلم والأداء")
        print("⚡ اختبار شامل للتأكد من فعالية الاستبدال الثوري")
        print("🌟" + "="*80 + "🌟")

        self.test_results = {}
        self.performance_metrics = {}
        self.learning_curves = {}

    def test_expert_explorer_system(self) -> Dict[str, Any]:
        """اختبار نظام الخبير/المستكشف الثوري"""
        print("\n🧠 اختبار نظام الخبير/المستكشف الثوري...")

        if not REVOLUTIONARY_SYSTEMS_AVAILABLE:
            return {"status": "failed", "reason": "النظام غير متوفر"}

        try:
            # إنشاء إعدادات ثورية
            config = RevolutionaryLearningConfig(
                strategy=RevolutionaryLearningStrategy.BASIL_INTEGRATIVE,
                adaptation_rate=0.02,
                wisdom_accumulation_factor=0.95,
                exploration_curiosity=0.3
            )

            # إنشاء النظام الثوري
            system = RevolutionaryExpertExplorerSystem(config)

            # اختبار التعلم
            learning_progress = []
            wisdom_accumulation = []

            print("📊 بدء مراقبة التعلم...")
            for episode in range(50):
                # محاكاة موقف رياضي
                situation = np.random.rand(10)

                # اتخاذ قرار خبير
                decision = system.make_expert_decision(situation)

                # محاكاة تطور الموقف
                evolved_situation = situation + np.random.rand(10) * 0.1

                # حساب مكسب الحكمة
                wisdom_gain = np.random.rand() * 0.8 + 0.2

                # إنشاء تجربة ثورية
                experience = RevolutionaryExperience(
                    situation=situation,
                    expert_decision=decision,
                    wisdom_gain=wisdom_gain,
                    evolved_situation=evolved_situation,
                    completion_status=(episode % 10 == 9)
                )

                # التطور من الحكمة
                evolution_stats = system.evolve_from_wisdom(experience)

                # تسجيل التقدم
                learning_progress.append(evolution_stats.get('wisdom_evolution', 0))
                wisdom_accumulation.append(system.total_wisdom_accumulated)

                if episode % 10 == 0:
                    print(f"  📈 الحلقة {episode}: حكمة متراكمة = {system.total_wisdom_accumulated:.3f}")

            # تحليل النتائج
            final_wisdom = wisdom_accumulation[-1]
            learning_improvement = (wisdom_accumulation[-1] - wisdom_accumulation[0]) / max(wisdom_accumulation[0], 0.001)

            result = {
                "status": "success",
                "final_wisdom": final_wisdom,
                "learning_improvement": learning_improvement,
                "wisdom_curve": wisdom_accumulation,
                "learning_detected": learning_improvement > 0.1,
                "system_evolved": len(set(learning_progress)) > 1
            }

            print(f"✅ النتيجة: الحكمة النهائية = {final_wisdom:.3f}")
            print(f"✅ تحسن التعلم: {learning_improvement*100:.1f}%")
            print(f"✅ هل يتعلم النظام؟ {'نعم' if result['learning_detected'] else 'لا'}")

            return result

        except Exception as e:
            print(f"❌ خطأ في الاختبار: {e}")
            return {"status": "error", "error": str(e)}

    def test_adaptive_equation_system(self) -> Dict[str, Any]:
        """اختبار نظام المعادلات المتكيفة الثوري"""
        print("\n🧮 اختبار نظام المعادلات المتكيفة الثوري...")

        if not REVOLUTIONARY_SYSTEMS_AVAILABLE:
            return {"status": "failed", "reason": "النظام غير متوفر"}

        try:
            # إنشاء إعدادات ثورية
            config = RevolutionaryAdaptiveConfig(
                adaptation_rate=0.015,
                wisdom_accumulation=0.96,
                exploration_curiosity=0.25
            )

            # إنشاء النظام الثوري
            system = RevolutionaryAdaptiveEquationSystem(config)

            # اختبار تطور المعادلات
            equation_evolution = []
            adaptation_progress = []

            print("📊 بدء مراقبة تطور المعادلات...")
            for iteration in range(30):
                # محاكاة موقف رياضي معقد
                situation = np.random.rand(15) * 2 - 1  # قيم بين -1 و 1

                # اتخاذ قرار بالمعادلة
                decision = system.make_equation_decision(situation)

                # محاكاة تطور
                evolved_situation = situation * 0.9 + np.random.rand(15) * 0.2

                # حساب مكسب الحكمة
                wisdom_gain = np.sum(np.abs(evolved_situation - situation))

                # إنشاء تجربة متكيفة
                experience = RevolutionaryAdaptiveExperience(
                    mathematical_situation=situation,
                    equation_decision=decision,
                    wisdom_gain=wisdom_gain,
                    evolved_situation=evolved_situation,
                    completion_status=(iteration % 5 == 4)
                )

                # تطور المعادلات
                evolution_stats = system.evolve_equations(experience)

                # تسجيل التطور
                equation_evolution.append(evolution_stats.get('equation_complexity', 0))
                adaptation_progress.append(evolution_stats.get('adaptation_strength', 0))

                if iteration % 5 == 0:
                    print(f"  🔬 التكرار {iteration}: تعقد المعادلة = {equation_evolution[-1]:.3f}")

            # تحليل التطور
            equation_improvement = (equation_evolution[-1] - equation_evolution[0]) / max(equation_evolution[0], 0.001)
            adaptation_variance = np.var(adaptation_progress)

            result = {
                "status": "success",
                "equation_improvement": equation_improvement,
                "adaptation_variance": adaptation_variance,
                "evolution_curve": equation_evolution,
                "equations_evolved": adaptation_variance > 0.01,
                "adaptive_learning": equation_improvement > 0.05
            }

            print(f"✅ تحسن المعادلات: {equation_improvement*100:.1f}%")
            print(f"✅ تنوع التكيف: {adaptation_variance:.4f}")
            print(f"✅ هل تتطور المعادلات؟ {'نعم' if result['equations_evolved'] else 'لا'}")

            return result

        except Exception as e:
            print(f"❌ خطأ في الاختبار: {e}")
            return {"status": "error", "error": str(e)}

    def test_revolutionary_agent(self) -> Dict[str, Any]:
        """اختبار الوكيل الثوري"""
        print("\n🤖 اختبار الوكيل الثوري...")

        if not REVOLUTIONARY_SYSTEMS_AVAILABLE:
            return {"status": "failed", "reason": "النظام غير متوفر"}

        try:
            # إنشاء إعدادات الوكيل الثوري
            config = RevolutionaryAgentConfig(
                decision_strategy=RevolutionaryDecisionStrategy.BASIL_INTEGRATIVE,
                adaptation_rate=0.02,
                wisdom_accumulation=0.94,
                exploration_curiosity=0.3
            )

            # إنشاء الوكيل الثوري
            agent = RevolutionaryExpertExplorerAgent(config, 12, 8)

            # اختبار اتخاذ القرارات والتعلم
            decision_quality = []
            wisdom_signals = []

            print("📊 بدء مراقبة أداء الوكيل...")
            for step in range(40):
                # محاكاة موقف
                situation = np.random.rand(12)

                # اتخاذ قرار
                decision = agent.make_revolutionary_decision(situation)

                # محاكاة نتيجة
                wisdom_gain = np.random.rand() * 0.9 + 0.1
                evolved_situation = situation + np.random.rand(12) * 0.15

                # التعلم من الحكمة
                learning_stats = agent.learn_from_wisdom(
                    situation, decision, wisdom_gain, evolved_situation,
                    step % 8 == 7
                )

                # تسجيل الأداء
                decision_quality.append(learning_stats.get('decision_quality', 0))
                wisdom_signals.append(learning_stats.get('wisdom_signal', 0))

                if step % 8 == 0:
                    print(f"  🎯 الخطوة {step}: جودة القرار = {decision_quality[-1]:.3f}")

            # تحليل الأداء
            quality_improvement = (decision_quality[-1] - decision_quality[0]) / max(decision_quality[0], 0.001)
            wisdom_consistency = 1 - np.var(wisdom_signals) / max(np.mean(wisdom_signals), 0.001)

            result = {
                "status": "success",
                "quality_improvement": quality_improvement,
                "wisdom_consistency": wisdom_consistency,
                "decision_curve": decision_quality,
                "agent_learning": quality_improvement > 0.1,
                "stable_wisdom": wisdom_consistency > 0.7
            }

            print(f"✅ تحسن جودة القرار: {quality_improvement*100:.1f}%")
            print(f"✅ ثبات الحكمة: {wisdom_consistency*100:.1f}%")
            print(f"✅ هل يتعلم الوكيل؟ {'نعم' if result['agent_learning'] else 'لا'}")

            return result

        except Exception as e:
            print(f"❌ خطأ في الاختبار: {e}")
            return {"status": "error", "error": str(e)}

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """تشغيل الاختبار الشامل"""
        print("\n🚀 بدء الاختبار الشامل للأنظمة الثورية...")

        # اختبار جميع الأنظمة
        expert_explorer_results = self.test_expert_explorer_system()
        adaptive_equation_results = self.test_adaptive_equation_system()
        revolutionary_agent_results = self.test_revolutionary_agent()

        # تجميع النتائج
        comprehensive_results = {
            "expert_explorer_system": expert_explorer_results,
            "adaptive_equation_system": adaptive_equation_results,
            "revolutionary_agent": revolutionary_agent_results,
            "overall_assessment": self._assess_overall_performance(
                expert_explorer_results, adaptive_equation_results, revolutionary_agent_results
            )
        }

        # طباعة التقييم الشامل
        self._print_comprehensive_assessment(comprehensive_results)

        return comprehensive_results

    def _assess_overall_performance(self, expert_results, equation_results, agent_results) -> Dict[str, Any]:
        """تقييم الأداء الشامل"""

        # حساب معدلات النجاح
        systems_working = sum([
            expert_results.get("status") == "success",
            equation_results.get("status") == "success",
            agent_results.get("status") == "success"
        ])

        # حساب معدلات التعلم
        learning_detected = sum([
            expert_results.get("learning_detected", False),
            equation_results.get("adaptive_learning", False),
            agent_results.get("agent_learning", False)
        ])

        # حساب القفزة في الأداء
        performance_improvements = [
            expert_results.get("learning_improvement", 0),
            equation_results.get("equation_improvement", 0),
            agent_results.get("quality_improvement", 0)
        ]

        avg_improvement = np.mean([imp for imp in performance_improvements if imp > 0])

        return {
            "systems_working": systems_working,
            "total_systems": 3,
            "learning_systems": learning_detected,
            "average_improvement": avg_improvement,
            "revolutionary_success": systems_working >= 2 and learning_detected >= 2,
            "performance_leap": avg_improvement > 0.2,
            "overall_score": (systems_working * 0.4 + learning_detected * 0.6) / 3.0
        }

    def _print_comprehensive_assessment(self, results: Dict[str, Any]):
        """طباعة التقييم الشامل"""
        print("\n" + "🌟" + "="*80 + "🌟")
        print("📊 التقييم الشامل للأنظمة الثورية")
        print("🌟" + "="*80 + "🌟")

        overall = results["overall_assessment"]

        print(f"🔧 الأنظمة العاملة: {overall['systems_working']}/{overall['total_systems']}")
        print(f"🧠 الأنظمة المتعلمة: {overall['learning_systems']}/{overall['total_systems']}")
        print(f"📈 متوسط التحسن: {overall['average_improvement']*100:.1f}%")
        print(f"🚀 نجاح ثوري: {'نعم' if overall['revolutionary_success'] else 'لا'}")
        print(f"⚡ قفزة في الأداء: {'نعم' if overall['performance_leap'] else 'لا'}")
        print(f"🏆 النتيجة الإجمالية: {overall['overall_score']*100:.1f}%")

        # تقييم نهائي
        if overall['overall_score'] >= 0.8:
            print("\n🎉 ممتاز! الاستبدال الثوري حقق نجاح<|im_start|> باهر!")
        elif overall['overall_score'] >= 0.6:
            print("\n✅ جيد! الاستبدال الثوري يعمل بفعالية!")
        elif overall['overall_score'] >= 0.4:
            print("\n⚠️ مقبول! الاستبدال يحتاج تحسينات!")
        else:
            print("\n❌ يحتاج عمل! الاستبدال يحتاج مراجعة!")


if __name__ == "__main__":
    # تشغيل الاختبار الشامل
    tester = RevolutionarySystemTester()
    results = tester.run_comprehensive_test()

    # حفظ النتائج
    with open("revolutionary_systems_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 تم حفظ النتائج في: revolutionary_systems_test_results.json")
