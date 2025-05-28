#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبار تكامل طبقة التفكير الفيزيائي مع النظام الكامل

هذا الملف يتحقق من تكامل طبقة التفكير الفيزيائي مع باقي مكونات نظام بصيرة،
ويختبر التفاعل بين الطبقات المختلفة والوظائف المتكاملة.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import time

# إعداد مسارات الاستيراد
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد مكونات طبقة التفكير الفيزيائي
from physical_reasoning.core.physical_reasoning_engine import (
    PhysicalReasoningEngine, PhysicalConcept, Hypothesis, Theory
)
from physical_reasoning.core.contradiction_detector import (
    ContradictionDetector, ContradictionType, Contradiction
)
from physical_reasoning.core.hypothesis_testing import (
    HypothesisTester, TestResult, TestType
)
from physical_reasoning.core.layer_integration import (
    LayerIntegrationManager, IntegrationType
)

# استيراد مكونات النواة الرياضياتية
from mathematical_core.enhanced.general_shape_equation import (
    GeneralShapeEquation, DeepLearningAdapter, ReinforcementLearningAdapter
)
from mathematical_core.enhanced.expert_explorer_interaction import (
    ExpertExplorerSystem, ExplorationStrategy
)
from mathematical_core.enhanced.semantic_integration import (
    SemanticIntegrator, SemanticMapping
)

# استيراد مكونات المعالجة الرمزية
from symbolic_processing.symbolic_interpreter import (
    SymbolicInterpreter, SymbolicMapping
)

# استيراد مكونات التمثيل المعرفي
from knowledge_representation.cognitive_objects import (
    ConceptualGraph, ConceptNode, ConceptRelation
)

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integration_test')


class IntegrationTester:
    """فئة لاختبار تكامل طبقة التفكير الفيزيائي مع النظام الكامل."""

    def __init__(self):
        """تهيئة المختبر."""
        self.logger = logging.getLogger('integration_test.tester')
        self.logger.info("Initializing Integration Tester...")
        
        # تهيئة مكونات النظام
        self.init_components()
        
        # تهيئة بيانات الاختبار
        self.init_test_data()
        
        self.logger.info("Integration Tester initialized successfully.")

    def init_components(self):
        """تهيئة مكونات النظام المختلفة."""
        # تهيئة طبقة التفكير الفيزيائي
        self.physical_engine = PhysicalReasoningEngine()
        self.contradiction_detector = ContradictionDetector(self.physical_engine)
        self.hypothesis_tester = HypothesisTester(self.physical_engine)
        
        # تهيئة النواة الرياضياتية
        self.shape_equation = GeneralShapeEquation()
        self.dl_adapter = DeepLearningAdapter(self.shape_equation)
        self.rl_adapter = ReinforcementLearningAdapter(self.shape_equation)
        self.expert_explorer = ExpertExplorerSystem(self.shape_equation)
        self.semantic_integrator = SemanticIntegrator()
        
        # تهيئة المعالجة الرمزية
        self.symbolic_interpreter = SymbolicInterpreter()
        
        # تهيئة التمثيل المعرفي
        self.conceptual_graph = ConceptualGraph()
        
        # تهيئة مدير تكامل الطبقات
        self.layer_manager = LayerIntegrationManager(
            physical_engine=self.physical_engine,
            shape_equation=self.shape_equation,
            symbolic_interpreter=self.symbolic_interpreter,
            conceptual_graph=self.conceptual_graph
        )

    def init_test_data(self):
        """تهيئة بيانات الاختبار."""
        # إنشاء مفاهيم فيزيائية للاختبار
        self.test_concepts = {
            "mass": PhysicalConcept(id="c1", name="Mass", definition="Property of matter resisting acceleration."),
            "space": PhysicalConcept(id="c2", name="Space", definition="Extension in which objects exist."),
            "gravity": PhysicalConcept(id="c3", name="Gravity", definition="Attraction between masses."),
            "filament": PhysicalConcept(id="c4", name="Filament", definition="Fundamental thread-like structure."),
            "zero_duality": PhysicalConcept(id="c5", name="Zero Duality", definition="Emergence of zero into two orthogonal opposites.")
        }
        
        # إنشاء فرضيات للاختبار
        self.test_hypotheses = {
            "h1": Hypothesis(id="h1", name="Mass attracts Mass", statement="Mass exerts an attractive force on other mass."),
            "h2": Hypothesis(id="h2", name="Space expands", statement="Space naturally expands and separates."),
            "h3": Hypothesis(id="h3", name="Mass and Space are opposites", statement="Mass and Space are orthogonal opposites."),
            "h4": Hypothesis(id="h4", name="Filaments form Mass", statement="Mass is formed by condensed filaments."),
            "h5": Hypothesis(id="h5", name="Gravity is tension", statement="Gravity is tension between dense mass and light space.")
        }
        
        # إنشاء نظريات للاختبار
        self.test_theories = {
            "zero_duality_theory": Theory(
                id="t1", 
                name="Zero Duality Theory", 
                hypotheses=[
                    self.test_hypotheses["h3"],
                    self.test_hypotheses["h4"],
                    self.test_hypotheses["h5"]
                ]
            ),
            "standard_gravity": Theory(
                id="t2", 
                name="Standard Gravity Theory", 
                hypotheses=[
                    self.test_hypotheses["h1"]
                ]
            )
        }
        
        # إضافة المفاهيم والفرضيات إلى محرك التفكير الفيزيائي
        for concept in self.test_concepts.values():
            self.physical_engine.add_concept(concept)
        
        for hypothesis in self.test_hypotheses.values():
            self.physical_engine.add_hypothesis(hypothesis)
        
        for theory in self.test_theories.values():
            self.physical_engine.add_theory(theory)
        
        # إنشاء عقد مفاهيم للرسم البياني المفاهيمي
        for concept_id, concept in self.test_concepts.items():
            node = ConceptNode(
                id=concept_id,
                name=concept.name,
                description=concept.definition
            )
            self.conceptual_graph.add_node(node)
        
        # إنشاء علاقات بين المفاهيم
        relations = [
            ConceptRelation(source_id="c1", target_id="c3", relation_type="causes", weight=1.0),
            ConceptRelation(source_id="c1", target_id="c2", relation_type="opposite_to", weight=1.0),
            ConceptRelation(source_id="c4", target_id="c1", relation_type="forms", weight=1.0),
            ConceptRelation(source_id="c5", target_id="c1", relation_type="explains", weight=0.8),
            ConceptRelation(source_id="c5", target_id="c2", relation_type="explains", weight=0.8)
        ]
        
        for relation in relations:
            self.conceptual_graph.add_relation(relation)

    def run_all_tests(self):
        """تشغيل جميع اختبارات التكامل."""
        self.logger.info("Starting integration tests...")
        
        test_methods = [
            self.test_physical_mathematical_integration,
            self.test_physical_symbolic_integration,
            self.test_physical_conceptual_integration,
            self.test_contradiction_detection_integration,
            self.test_hypothesis_testing_integration,
            self.test_end_to_end_theory_analysis
        ]
        
        results = {}
        all_passed = True
        
        for test_method in test_methods:
            test_name = test_method.__name__
            self.logger.info(f"Running test: {test_name}")
            
            try:
                passed = test_method()
                results[test_name] = "PASSED" if passed else "FAILED"
                if not passed:
                    all_passed = False
            except Exception as e:
                self.logger.error(f"Error in test {test_name}: {str(e)}")
                results[test_name] = f"ERROR: {str(e)}"
                all_passed = False
        
        self.logger.info("Integration tests completed.")
        self.logger.info("Results:")
        for test_name, result in results.items():
            self.logger.info(f"  {test_name}: {result}")
        
        return all_passed, results

    def test_physical_mathematical_integration(self):
        """اختبار تكامل طبقة التفكير الفيزيائي مع النواة الرياضياتية."""
        self.logger.info("Testing integration between Physical Reasoning and Mathematical Core...")
        
        try:
            # اختبار تحويل فرضية فيزيائية إلى معادلة شكل عام
            hypothesis = self.test_hypotheses["h5"]  # فرضية الجاذبية كشد
            
            # تحويل الفرضية إلى معادلة
            equation = self.layer_manager.convert_hypothesis_to_equation(hypothesis)
            
            # التحقق من نجاح التحويل
            if equation is None:
                self.logger.error("Failed to convert hypothesis to equation")
                return False
            
            self.logger.info(f"Successfully converted hypothesis '{hypothesis.name}' to equation: {equation}")
            
            # اختبار استخدام معادلة الشكل العام لتوليد تنبؤات
            predictions = self.layer_manager.generate_predictions_from_equation(equation, {"scenario": "apple_falls"})
            
            # التحقق من وجود تنبؤات
            if not predictions:
                self.logger.error("Failed to generate predictions from equation")
                return False
            
            self.logger.info(f"Successfully generated predictions from equation: {predictions}")
            
            # اختبار تكامل نظام الخبير/المستكشف
            exploration_results = self.expert_explorer.explore_equation_space(
                base_equation=equation,
                strategy=ExplorationStrategy.SEMANTIC_GUIDED,
                iterations=5
            )
            
            # التحقق من نجاح الاستكشاف
            if not exploration_results:
                self.logger.error("Failed to explore equation space")
                return False
            
            self.logger.info(f"Successfully explored equation space: {len(exploration_results)} variations found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in physical-mathematical integration test: {str(e)}")
            return False

    def test_physical_symbolic_integration(self):
        """اختبار تكامل طبقة التفكير الفيزيائي مع المعالجة الرمزية."""
        self.logger.info("Testing integration between Physical Reasoning and Symbolic Processing...")
        
        try:
            # اختبار تحويل مفهوم فيزيائي إلى تمثيل رمزي
            concept = self.test_concepts["filament"]  # مفهوم الفتيلة
            
            # تحويل المفهوم إلى تمثيل رمزي
            symbolic_rep = self.layer_manager.convert_concept_to_symbolic(concept)
            
            # التحقق من نجاح التحويل
            if symbolic_rep is None:
                self.logger.error("Failed to convert concept to symbolic representation")
                return False
            
            self.logger.info(f"Successfully converted concept '{concept.name}' to symbolic representation: {symbolic_rep}")
            
            # اختبار تحويل فرضية فيزيائية إلى تمثيل رمزي
            hypothesis = self.test_hypotheses["h4"]  # فرضية تشكل الكتلة من الفتائل
            
            # تحويل الفرضية إلى تمثيل رمزي
            symbolic_hypo = self.layer_manager.convert_hypothesis_to_symbolic(hypothesis)
            
            # التحقق من نجاح التحويل
            if symbolic_hypo is None:
                self.logger.error("Failed to convert hypothesis to symbolic representation")
                return False
            
            self.logger.info(f"Successfully converted hypothesis '{hypothesis.name}' to symbolic representation: {symbolic_hypo}")
            
            # اختبار تفسير التمثيل الرمزي
            interpretation = self.symbolic_interpreter.interpret(symbolic_hypo)
            
            # التحقق من نجاح التفسير
            if not interpretation:
                self.logger.error("Failed to interpret symbolic representation")
                return False
            
            self.logger.info(f"Successfully interpreted symbolic representation: {interpretation}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in physical-symbolic integration test: {str(e)}")
            return False

    def test_physical_conceptual_integration(self):
        """اختبار تكامل طبقة التفكير الفيزيائي مع التمثيل المعرفي."""
        self.logger.info("Testing integration between Physical Reasoning and Knowledge Representation...")
        
        try:
            # اختبار تحويل نظرية فيزيائية إلى رسم بياني مفاهيمي
            theory = self.test_theories["zero_duality_theory"]
            
            # تحويل النظرية إلى رسم بياني مفاهيمي
            graph_update = self.layer_manager.convert_theory_to_conceptual_graph(theory, self.conceptual_graph)
            
            # التحقق من نجاح التحويل
            if not graph_update:
                self.logger.error("Failed to convert theory to conceptual graph")
                return False
            
            self.logger.info(f"Successfully converted theory '{theory.name}' to conceptual graph")
            
            # اختبار استخراج المفاهيم المرتبطة من الرسم البياني
            related_concepts = self.conceptual_graph.get_related_nodes("c5")  # المفاهيم المرتبطة بالانبثاق الثنائي للصفر
            
            # التحقق من وجود مفاهيم مرتبطة
            if not related_concepts:
                self.logger.error("Failed to retrieve related concepts from graph")
                return False
            
            self.logger.info(f"Successfully retrieved {len(related_concepts)} related concepts from graph")
            
            # اختبار تنشيط المفاهيم في الرسم البياني
            self.conceptual_graph.activate_node("c5", 1.0, propagate=True)
            activated_nodes = self.conceptual_graph.get_activated_nodes(threshold=0.1)
            
            # التحقق من وجود عقد منشطة
            if not activated_nodes:
                self.logger.error("Failed to activate and propagate through conceptual graph")
                return False
            
            self.logger.info(f"Successfully activated and propagated through graph: {len(activated_nodes)} nodes activated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in physical-conceptual integration test: {str(e)}")
            return False

    def test_contradiction_detection_integration(self):
        """اختبار تكامل آلية اكتشاف التناقضات مع النظام."""
        self.logger.info("Testing integration of Contradiction Detection...")
        
        try:
            # إنشاء فرضيات متناقضة للاختبار
            contradictory_hypothesis = Hypothesis(
                id="h_contra", 
                name="Space contracts", 
                statement="Space naturally contracts and condenses."
            )
            
            # إضافة الفرضية المتناقضة إلى محرك التفكير الفيزيائي
            self.physical_engine.add_hypothesis(contradictory_hypothesis)
            
            # اختبار اكتشاف التناقضات بين الفرضيات
            elements_to_check = [
                self.test_hypotheses["h2"],  # فرضية تمدد المكان
                contradictory_hypothesis  # فرضية تقلص المكان
            ]
            
            contradictions = self.contradiction_detector.detect_contradictions(
                elements=elements_to_check,
                types_to_detect=[ContradictionType.LOGICAL, ContradictionType.PREDICTIVE]
            )
            
            # التحقق من اكتشاف تناقضات
            if not contradictions:
                self.logger.error("Failed to detect contradictions between opposing hypotheses")
                return False
            
            self.logger.info(f"Successfully detected {len(contradictions)} contradictions")
            
            # اختبار تكامل اكتشاف التناقضات مع الرسم البياني المفاهيمي
            # تحويل التناقضات إلى علاقات في الرسم البياني
            for contradiction in contradictions:
                relation = ConceptRelation(
                    source_id="h2",
                    target_id="h_contra",
                    relation_type="contradicts",
                    weight=contradiction.confidence,
                    properties={"contradiction_type": contradiction.type.name}
                )
                self.conceptual_graph.add_relation(relation)
            
            # التحقق من إضافة العلاقات
            contradictory_relations = [rel for rel in self.conceptual_graph.get_relations() if rel.relation_type == "contradicts"]
            
            if not contradictory_relations:
                self.logger.error("Failed to add contradiction relations to conceptual graph")
                return False
            
            self.logger.info(f"Successfully added {len(contradictory_relations)} contradiction relations to graph")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in contradiction detection integration test: {str(e)}")
            return False

    def test_hypothesis_testing_integration(self):
        """اختبار تكامل آلية اختبار الفرضيات مع النظام."""
        self.logger.info("Testing integration of Hypothesis Testing...")
        
        try:
            # اختبار فرضية باستخدام بيانات تجريبية وهمية
            hypothesis = self.test_hypotheses["h5"]  # فرضية الجاذبية كشد
            
            # إنشاء بيانات تجريبية وهمية
            experimental_data = {
                "apple_falls": True,
                "planets_orbit": True,
                "galaxies_form": True
            }
            
            # اختبار الفرضية
            test_results = self.hypothesis_tester.test_hypothesis(
                hypothesis=hypothesis,
                test_types=[TestType.EMPIRICAL, TestType.LOGICAL, TestType.PREDICTIVE],
                context={"experimental_data": experimental_data}
            )
            
            # التحقق من وجود نتائج اختبار
            if not test_results:
                self.logger.error("Failed to test hypothesis")
                return False
            
            self.logger.info(f"Successfully tested hypothesis with {len(test_results)} test results")
            
            # اختبار تكامل نتائج الاختبار مع النواة الرياضياتية
            # تحويل نتائج الاختبار إلى معلمات لتحسين المعادلة
            equation = self.layer_manager.convert_hypothesis_to_equation(hypothesis)
            
            if equation is None:
                self.logger.error("Failed to convert hypothesis to equation for parameter tuning")
                return False
            
            # تحسين المعادلة بناءً على نتائج الاختبار
            improved_equation = self.layer_manager.tune_equation_parameters(
                equation=equation,
                test_results=test_results
            )
            
            # التحقق من تحسين المعادلة
            if improved_equation is None:
                self.logger.error("Failed to tune equation parameters based on test results")
                return False
            
            self.logger.info("Successfully tuned equation parameters based on test results")
            
            # اختبار تكامل نتائج الاختبار مع التمثيل المعرفي
            # تحديث الثقة في الفرضية في الرسم البياني المفاهيمي
            overall_confidence = sum(result.confidence for result in test_results) / len(test_results)
            
            node = self.conceptual_graph.get_node(hypothesis.id)
            if node:
                node.properties["confidence"] = overall_confidence
                self.logger.info(f"Successfully updated hypothesis confidence in conceptual graph: {overall_confidence}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in hypothesis testing integration test: {str(e)}")
            return False

    def test_end_to_end_theory_analysis(self):
        """اختبار تحليل نظرية من البداية إلى النهاية."""
        self.logger.info("Testing end-to-end theory analysis...")
        
        try:
            # اختيار نظرية للتحليل
            theory = self.test_theories["zero_duality_theory"]
            
            # 1. تحويل النظرية إلى تمثيلات مختلفة
            # 1.1 تحويل إلى معادلات
            equations = {}
            for hypothesis in theory.hypotheses:
                equation = self.layer_manager.convert_hypothesis_to_equation(hypothesis)
                if equation:
                    equations[hypothesis.id] = equation
            
            # التحقق من تحويل المعادلات
            if not equations:
                self.logger.error("Failed to convert theory hypotheses to equations")
                return False
            
            self.logger.info(f"Successfully converted {len(equations)} hypotheses to equations")
            
            # 1.2 تحويل إلى تمثيلات رمزية
            symbolic_reps = {}
            for hypothesis in theory.hypotheses:
                symbolic_rep = self.layer_manager.convert_hypothesis_to_symbolic(hypothesis)
                if symbolic_rep:
                    symbolic_reps[hypothesis.id] = symbolic_rep
            
            # التحقق من التمثيلات الرمزية
            if not symbolic_reps:
                self.logger.error("Failed to convert theory hypotheses to symbolic representations")
                return False
            
            self.logger.info(f"Successfully converted {len(symbolic_reps)} hypotheses to symbolic representations")
            
            # 1.3 تحويل إلى رسم بياني مفاهيمي
            graph_update = self.layer_manager.convert_theory_to_conceptual_graph(theory, self.conceptual_graph)
            
            # التحقق من تحديث الرسم البياني
            if not graph_update:
                self.logger.error("Failed to convert theory to conceptual graph")
                return False
            
            self.logger.info("Successfully converted theory to conceptual graph")
            
            # 2. اكتشاف التناقضات الداخلية
            internal_contradictions = self.contradiction_detector.detect_contradictions(
                elements=[theory],
                types_to_detect=[ContradictionType.INTERNAL]
            )
            
            self.logger.info(f"Detected {len(internal_contradictions)} internal contradictions in theory")
            
            # 3. اختبار الفرضيات
            test_results = {}
            for hypothesis in theory.hypotheses:
                results = self.hypothesis_tester.test_hypothesis(
                    hypothesis=hypothesis,
                    test_types=[TestType.LOGICAL, TestType.CONCEPTUAL]
                )
                if results:
                    test_results[hypothesis.id] = results
            
            # التحقق من نتائج الاختبار
            if not test_results:
                self.logger.error("Failed to test theory hypotheses")
                return False
            
            self.logger.info(f"Successfully tested {len(test_results)} hypotheses")
            
            # 4. تحسين النظرية
            # 4.1 تحسين المعادلات
            improved_equations = {}
            for hypo_id, equation in equations.items():
                if hypo_id in test_results:
                    improved_eq = self.layer_manager.tune_equation_parameters(
                        equation=equation,
                        test_results=test_results[hypo_id]
                    )
                    if improved_eq:
                        improved_equations[hypo_id] = improved_eq
            
            # التحقق من تحسين المعادلات
            if not improved_equations:
                self.logger.error("Failed to improve equations based on test results")
                return False
            
            self.logger.info(f"Successfully improved {len(improved_equations)} equations")
            
            # 4.2 استكشاف فرضيات جديدة
            new_hypotheses = []
            for hypo_id, equation in improved_equations.items():
                variations = self.expert_explorer.explore_equation_space(
                    base_equation=equation,
                    strategy=ExplorationStrategy.SEMANTIC_GUIDED,
                    iterations=3
                )
                
                for i, var_eq in enumerate(variations[:2]):  # استخدام أول تنويعين فقط
                    new_hypo = self.layer_manager.convert_equation_to_hypothesis(
                        equation=var_eq,
                        base_hypothesis_id=hypo_id,
                        suffix=f"_var{i+1}"
                    )
                    if new_hypo:
                        new_hypotheses.append(new_hypo)
            
            # التحقق من الفرضيات الجديدة
            if not new_hypotheses:
                self.logger.error("Failed to generate new hypotheses through exploration")
                return False
            
            self.logger.info(f"Successfully generated {len(new_hypotheses)} new hypotheses through exploration")
            
            # 5. إنشاء نظرية محسنة
            improved_theory = Theory(
                id=f"{theory.id}_improved",
                name=f"{theory.name} (Improved)",
                hypotheses=theory.hypotheses + new_hypotheses
            )
            
            # إضافة النظرية المحسنة إلى محرك التفكير الفيزيائي
            self.physical_engine.add_theory(improved_theory)
            
            self.logger.info(f"Successfully created improved theory: {improved_theory.name}")
            
            # 6. تقييم النظرية المحسنة
            # 6.1 اكتشاف التناقضات الداخلية
            improved_contradictions = self.contradiction_detector.detect_contradictions(
                elements=[improved_theory],
                types_to_detect=[ContradictionType.INTERNAL]
            )
            
            self.logger.info(f"Detected {len(improved_contradictions)} internal contradictions in improved theory")
            
            # 6.2 مقارنة النظريتين
            comparison = self.layer_manager.compare_theories(
                theory1=theory,
                theory2=improved_theory
            )
            
            # التحقق من المقارنة
            if not comparison:
                self.logger.error("Failed to compare original and improved theories")
                return False
            
            self.logger.info("Successfully compared original and improved theories")
            self.logger.info(f"Comparison results: {comparison}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in end-to-end theory analysis test: {str(e)}")
            return False


# --- تشغيل الاختبارات --- #
if __name__ == '__main__':
    tester = IntegrationTester()
    all_passed, results = tester.run_all_tests()
    
    if all_passed:
        print("\n✅ All integration tests passed successfully!")
    else:
        print("\n❌ Some integration tests failed. See log for details.")
    
    print("\nTest Results:")
    for test_name, result in results.items():
        print(f"  {'✅' if result == 'PASSED' else '❌'} {test_name}: {result}")
