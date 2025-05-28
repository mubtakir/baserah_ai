#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
آليات اكتشاف التناقضات في طبقة التفكير الفيزيائي

هذا الملف يحتوي على الفئات والوظائف المسؤولة عن اكتشاف التناقضات
المنطقية والمفاهيمية والتنبؤية بين الفرضيات والنظريات.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
from typing import List, Tuple, Dict, Optional, Any, Set, Union
from enum import Enum, auto
import logging
from dataclasses import dataclass, field

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from physical_reasoning_engine import (
    PhysicalConcept, Hypothesis, Theory, PhysicalReasoningEngine
)

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('physical_reasoning.contradiction_detector')


class ContradictionType(Enum):
    """أنواع التناقضات."""
    LOGICAL = auto()  # تناقض منطقي (مثل A و not A)
    CONCEPTUAL = auto()  # تناقض مفاهيمي (مثل تعريفين متعارضين لنفس المفهوم)
    PREDICTIVE = auto()  # تناقض تنبؤي (مثل تنبؤين متعارضين لنفس الظاهرة)
    EVIDENTIAL = auto()  # تناقض مع الأدلة (مثل فرضية تتعارض مع دليل تجريبي)
    AXIOMATIC = auto()  # تناقض مع البديهيات (مثل فرضية تتعارض مع بديهية أساسية)
    INTERNAL = auto()  # تناقض داخلي (مثل فرضية تتناقض مع نفسها أو مع فرضية أخرى في نفس النظرية)
    EXTERNAL = auto()  # تناقض خارجي (مثل فرضية تتناقض مع فرضية في نظرية أخرى مقبولة)


@dataclass
class Contradiction:
    """يمثل تناقضاً مكتشفاً."""
    type: ContradictionType  # نوع التناقض
    elements: List[Union[Hypothesis, Theory, PhysicalConcept, Any]]  # العناصر المتناقضة
    description: str  # وصف التناقض
    confidence: float = 1.0  # مستوى الثقة في وجود التناقض
    metadata: Dict[str, Any] = field(default_factory=dict)  # بيانات وصفية إضافية

    def __str__(self) -> str:
        """تمثيل نصي للتناقض."""
        element_names = [getattr(el, 'name', str(el)) for el in self.elements]
        return f"Contradiction ({self.type.name}): {self.description} involving {element_names} (Confidence: {self.confidence:.2f})"


class ContradictionDetector:
    """فئة لاكتشاف التناقضات بين الفرضيات والنظريات."""

    def __init__(self, reasoning_engine: Optional[PhysicalReasoningEngine] = None):
        """تهيئة كاشف التناقضات."""
        self.reasoning_engine = reasoning_engine
        self.logger = logging.getLogger('physical_reasoning.contradiction_detector')
        self.detection_methods = {
            ContradictionType.LOGICAL: self._detect_logical_contradictions,
            ContradictionType.CONCEPTUAL: self._detect_conceptual_contradictions,
            ContradictionType.PREDICTIVE: self._detect_predictive_contradictions,
            ContradictionType.EVIDENTIAL: self._detect_evidential_contradictions,
            ContradictionType.AXIOMATIC: self._detect_axiomatic_contradictions,
            ContradictionType.INTERNAL: self._detect_internal_contradictions,
            ContradictionType.EXTERNAL: self._detect_external_contradictions,
        }

    def detect_contradictions(self, elements: List[Union[Hypothesis, Theory, PhysicalConcept]],
                              context: Optional[Dict[str, Any]] = None,
                              types_to_detect: Optional[List[ContradictionType]] = None) -> List[Contradiction]:
        """
        اكتشاف التناقضات بين مجموعة من العناصر.

        Args:
            elements: قائمة بالفرضيات والنظريات والمفاهيم للتحقق منها.
            context: سياق إضافي (مثل الأدلة المتاحة، النظريات المقبولة).
            types_to_detect: قائمة بأنواع التناقضات المراد اكتشافها (افتراضي: الكل).

        Returns:
            قائمة بالتناقضات المكتشفة.
        """
        contradictions: List[Contradiction] = []
        context = context or {}

        if types_to_detect is None:
            types_to_detect = list(ContradictionType)

        self.logger.info(f"Starting contradiction detection for {len(elements)} elements.")

        # مقارنة كل زوج من العناصر (للتناقضات الخارجية)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                el1 = elements[i]
                el2 = elements[j]
                for contradiction_type in types_to_detect:
                    if contradiction_type in self.detection_methods:
                        method = self.detection_methods[contradiction_type]
                        found = method(el1, el2, context)
                        contradictions.extend(found)

        # فحص التناقضات الداخلية لكل عنصر (إذا كان نظرية)
        for element in elements:
            if isinstance(element, Theory) and ContradictionType.INTERNAL in types_to_detect:
                 found = self._detect_internal_contradictions(element, element, context) # Pass element twice for internal check
                 contradictions.extend(found)
            # فحص التناقضات مع الأدلة والبديهيات لكل عنصر
            if ContradictionType.EVIDENTIAL in types_to_detect:
                 found = self._detect_evidential_contradictions(element, None, context)
                 contradictions.extend(found)
            if ContradictionType.AXIOMATIC in types_to_detect:
                 found = self._detect_axiomatic_contradictions(element, None, context)
                 contradictions.extend(found)


        self.logger.info(f"Found {len(contradictions)} contradictions.")
        return contradictions

    def _detect_logical_contradictions(self, el1: Any, el2: Any, context: Dict[str, Any]) -> List[Contradiction]:
        """اكتشاف التناقضات المنطقية."""
        # مثال بسيط: التحقق مما إذا كانت فرضية تنص على P وأخرى تنص على not P
        # يتطلب تمثيل منطقي للفرضيات (مثل استخدام المنطق الرسمي أو مكتبات المنطق)
        contradictions = []
        # TODO: Implement more sophisticated logical contradiction detection
        # This requires a formal logic representation of hypotheses/theories
        if isinstance(el1, Hypothesis) and isinstance(el2, Hypothesis):
             # Placeholder: Check if one hypothesis statement is the negation of the other
             # This is highly simplified and needs a proper logic engine
             statement1 = el1.statement.lower().strip()
             statement2 = el2.statement.lower().strip()
             if statement1 == f"not {statement2}" or statement2 == f"not {statement1}":
                 contradictions.append(Contradiction(
                     type=ContradictionType.LOGICAL,
                     elements=[el1, el2],
                     description=f"Hypothesis '{el1.name}' states '{el1.statement}' while '{el2.name}' states '{el2.statement}', which are logical negations.",
                     confidence=0.9
                 ))
        return contradictions

    def _detect_conceptual_contradictions(self, el1: Any, el2: Any, context: Dict[str, Any]) -> List[Contradiction]:
        """اكتشاف التناقضات المفاهيمية."""
        # مثال: التحقق مما إذا كان تعريف مفهوم في نظرية يتعارض مع تعريفه في نظرية أخرى
        contradictions = []
        concepts1 = self._extract_concepts(el1)
        concepts2 = self._extract_concepts(el2)

        common_concept_ids = set(concepts1.keys()) & set(concepts2.keys())

        for concept_id in common_concept_ids:
            concept1 = concepts1[concept_id]
            concept2 = concepts2[concept_id]

            # قارن التعريفات أو الخصائص الأساسية
            # يتطلب طريقة موحدة لتمثيل التعريفات والمقارنة بينها
            # Placeholder: Simple description comparison
            if concept1.definition.strip().lower() != concept2.definition.strip().lower():
                 # Need a more robust semantic comparison here
                 similarity_threshold = 0.5 # Example threshold
                 # semantic_similarity = calculate_semantic_similarity(concept1.definition, concept2.definition)
                 # if semantic_similarity < similarity_threshold:
                 if True: # Simplified check
                    contradictions.append(Contradiction(
                        type=ContradictionType.CONCEPTUAL,
                        elements=[concept1, concept2],
                        description=f"Definition of concept '{concept1.name}' in '{getattr(el1, 'name', 'element1')}' ('{concept1.definition}') potentially conflicts with definition in '{getattr(el2, 'name', 'element2')}' ('{concept2.definition}').",
                        confidence=0.7
                    ))
        return contradictions

    def _detect_predictive_contradictions(self, el1: Any, el2: Any, context: Dict[str, Any]) -> List[Contradiction]:
        """اكتشاف التناقضات التنبؤية."""
        # مثال: التحقق مما إذا كانت نظريتان تتنبآن بنتائج متعارضة لنفس التجربة
        contradictions = []
        predictions1 = self._extract_predictions(el1)
        predictions2 = self._extract_predictions(el2)

        common_scenarios = set(predictions1.keys()) & set(predictions2.keys())

        for scenario in common_scenarios:
            pred1 = predictions1[scenario]
            pred2 = predictions2[scenario]

            # قارن التنبؤات (قد تكون قيم عددية، حالات، إلخ)
            # يتطلب طريقة موحدة لتمثيل التنبؤات والمقارنة بينها
            # Placeholder: Simple equality check for opposite outcomes
            if pred1 != pred2: # Needs refinement for non-binary/quantitative predictions
                 # Example: Check for opposite boolean predictions
                 if isinstance(pred1, bool) and isinstance(pred2, bool) and pred1 != pred2:
                     outcome1 = "True" if pred1 else "False"
                     outcome2 = "True" if pred2 else "False"
                     contradictions.append(Contradiction(
                         type=ContradictionType.PREDICTIVE,
                         elements=[el1, el2],
                         description=f"Conflicting predictions for scenario '{scenario}'. '{getattr(el1, 'name', 'element1')}' predicts {outcome1}, while '{getattr(el2, 'name', 'element2')}' predicts {outcome2}.",
                         confidence=0.95
                     ))
                 # Example: Check for significantly different quantitative predictions
                 elif isinstance(pred1, (int, float)) and isinstance(pred2, (int, float)):
                     diff = abs(pred1 - pred2)
                     avg = (abs(pred1) + abs(pred2)) / 2
                     relative_diff = diff / avg if avg > 1e-6 else diff
                     if relative_diff > 0.5: # Example threshold for significant difference
                         contradictions.append(Contradiction(
                             type=ContradictionType.PREDICTIVE,
                             elements=[el1, el2],
                             description=f"Significantly different quantitative predictions for scenario '{scenario}'. '{getattr(el1, 'name', 'element1')}' predicts {pred1}, while '{getattr(el2, 'name', 'element2')}' predicts {pred2}.",
                             confidence=0.8
                         ))

        return contradictions

    def _detect_evidential_contradictions(self, element: Any, _: Any, context: Dict[str, Any]) -> List[Contradiction]:
        """اكتشاف التناقضات مع الأدلة."""
        contradictions = []
        evidence = context.get('evidence', []) # قائمة بالأدلة التجريبية
        predictions = self._extract_predictions(element)

        for ev in evidence:
            scenario = ev.get('scenario')
            observed_outcome = ev.get('outcome')

            if scenario in predictions:
                predicted_outcome = predictions[scenario]
                # قارن التنبؤ مع الدليل
                if predicted_outcome != observed_outcome: # Needs refinement based on outcome types and uncertainty
                    contradictions.append(Contradiction(
                        type=ContradictionType.EVIDENTIAL,
                        elements=[element, ev],
                        description=f"Prediction of '{getattr(element, 'name', 'element')}' ({predicted_outcome}) for scenario '{scenario}' contradicts observed evidence ({observed_outcome}).",
                        confidence=0.9 # Confidence might depend on evidence quality
                    ))
        return contradictions

    def _detect_axiomatic_contradictions(self, element: Any, _: Any, context: Dict[str, Any]) -> List[Contradiction]:
        """اكتشاف التناقضات مع البديهيات."""
        contradictions = []
        axioms = context.get('axioms', []) # قائمة بالبديهيات الأساسية المقبولة
        assumptions = self._extract_assumptions(element)

        for assumption in assumptions:
            for axiom in axioms:
                # تحقق مما إذا كان الافتراض يتعارض منطقياً مع البديهية
                # Requires logical representation and inference engine
                # Placeholder check
                if assumption.statement.lower().strip() == f"not {axiom.statement.lower().strip()}":
                     contradictions.append(Contradiction(
                         type=ContradictionType.AXIOMATIC,
                         elements=[element, axiom],
                         description=f"Assumption '{assumption.statement}' within '{getattr(element, 'name', 'element')}' contradicts accepted axiom '{axiom.statement}'.",
                         confidence=1.0
                     ))
        return contradictions

    def _detect_internal_contradictions(self, element: Theory, _: Any, context: Dict[str, Any]) -> List[Contradiction]:
        """اكتشاف التناقضات الداخلية داخل نظرية واحدة."""
        contradictions = []
        if not isinstance(element, Theory):
            return []

        hypotheses = element.hypotheses
        # Compare all pairs of hypotheses within the theory
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                h1 = hypotheses[i]
                h2 = hypotheses[j]
                # Check for logical, conceptual, predictive contradictions between internal hypotheses
                found_logical = self._detect_logical_contradictions(h1, h2, context)
                for c in found_logical: c.type = ContradictionType.INTERNAL
                contradictions.extend(found_logical)

                found_conceptual = self._detect_conceptual_contradictions(h1, h2, context)
                for c in found_conceptual: c.type = ContradictionType.INTERNAL
                contradictions.extend(found_conceptual)

                found_predictive = self._detect_predictive_contradictions(h1, h2, context)
                for c in found_predictive: c.type = ContradictionType.INTERNAL
                contradictions.extend(found_predictive)

        # TODO: Check for contradictions between hypotheses and the theory's core assumptions/concepts

        return contradictions

    def _detect_external_contradictions(self, el1: Any, el2: Any, context: Dict[str, Any]) -> List[Contradiction]:
         """اكتشاف التناقضات الخارجية (بين عنصرين مختلفين)."""
         # This method essentially calls the other detection methods for pairs
         # It's separated for clarity in the main detection loop
         contradictions = []
         contradictions.extend(self._detect_logical_contradictions(el1, el2, context))
         contradictions.extend(self._detect_conceptual_contradictions(el1, el2, context))
         contradictions.extend(self._detect_predictive_contradictions(el1, el2, context))
         # Note: Evidential and Axiomatic checks are usually done per element against context,
         # but could be adapted for pairwise comparison if needed.
         return contradictions

    # --- Helper methods to extract relevant information --- #

    def _extract_concepts(self, element: Any) -> Dict[str, PhysicalConcept]:
        """استخراج المفاهيم من فرضية أو نظرية."""
        if isinstance(element, PhysicalConcept):
            return {element.id: element}
        elif isinstance(element, Hypothesis):
            # Assume concepts are linked or defined within the hypothesis properties
            return element.related_concepts or {}
        elif isinstance(element, Theory):
            concepts = {} # element.core_concepts or {}
            for hypo in element.hypotheses:
                concepts.update(self._extract_concepts(hypo))
            return concepts
        return {}

    def _extract_predictions(self, element: Any) -> Dict[str, Any]:
        """استخراج التنبؤات من فرضية أو نظرية."""
        if isinstance(element, Hypothesis):
            # Assume predictions are stored in properties or a dedicated field
            return element.predictions or {}
        elif isinstance(element, Theory):
            predictions = {} # element.global_predictions or {}
            for hypo in element.hypotheses:
                predictions.update(self._extract_predictions(hypo))
            # TODO: Resolve potential conflicts in predictions from different hypotheses within the theory
            return predictions
        return {}

    def _extract_assumptions(self, element: Any) -> List[Hypothesis]:
        """استخراج الافتراضات الأساسية من فرضية أو نظرية."""
        if isinstance(element, Hypothesis):
            # Assumptions might be explicitly listed or be the hypothesis itself
            return [element] # Simplified: treat the hypothesis statement as its core assumption
        elif isinstance(element, Theory):
            # Assumptions could be core hypotheses or explicitly defined
            return element.assumptions or element.hypotheses # Simplified
        return []


# --- مثال للاستخدام --- #
if __name__ == '__main__':
    # تعريف مفاهيم وفرضيات ونظريات وهمية
    concept_mass = PhysicalConcept(id="c1", name="Mass", definition="Property of matter resisting acceleration.")
    concept_spacetime = PhysicalConcept(id="c2", name="Spacetime", definition="Four-dimensional continuum.")
    concept_gravity_newton = PhysicalConcept(id="c3a", name="Gravity (Newton)", definition="Force between masses.")
    concept_gravity_einstein = PhysicalConcept(id="c3b", name="Gravity (Einstein)", definition="Curvature of spacetime caused by mass/energy.")

    hypo1 = Hypothesis(id="h1", name="H1: Mass attracts Mass", statement="Mass exerts an attractive force on other mass.", predictions={"apple_falls": True}, related_concepts={'c1': concept_mass, 'c3a': concept_gravity_newton})
    hypo2 = Hypothesis(id="h2", name="H2: Mass curves Spacetime", statement="Mass and energy curve spacetime.", predictions={"light_bends": True}, related_concepts={'c1': concept_mass, 'c2': concept_spacetime, 'c3b': concept_gravity_einstein})
    hypo3 = Hypothesis(id="h3", name="H3: No Action at a Distance", statement="Forces cannot act instantaneously over distance.", predictions={}, related_concepts={})
    hypo4 = Hypothesis(id="h4", name="H4: Action at a Distance", statement="Gravity acts instantaneously over distance.", predictions={}, related_concepts={})
    hypo5 = Hypothesis(id="h5", name="H5: Apple Floats", statement="Apples float upwards.", predictions={"apple_falls": False}, related_concepts={})


    theory_newton = Theory(id="t1", name="Newtonian Gravity", hypotheses=[hypo1, hypo4], core_concepts=[concept_mass, concept_gravity_newton])
    theory_einstein = Theory(id="t2", name="General Relativity", hypotheses=[hypo2, hypo3], core_concepts=[concept_mass, concept_spacetime, concept_gravity_einstein])
    theory_floating_apple = Theory(id="t3", name="Floating Apple Theory", hypotheses=[hypo5], core_concepts=[])

    # تعريف أدلة وبديهيات وهمية
    evidence_apple = {'scenario': 'apple_falls', 'outcome': True, 'source': 'Observation'}
    axiom_causality = Hypothesis(id="ax1", name="Axiom of Causality", statement="Effects have causes.") # Simplified representation

    context = {
        'evidence': [evidence_apple],
        'axioms': [axiom_causality]
    }

    # إنشاء كاشف التناقضات
    detector = ContradictionDetector()

    # اكتشاف التناقضات
    elements_to_check = [theory_newton, theory_einstein, theory_floating_apple, hypo1, hypo2, hypo3, hypo4, hypo5]
    all_contradictions = detector.detect_contradictions(elements_to_check, context)

    print("\nDetected Contradictions:")
    if all_contradictions:
        for contradiction in all_contradictions:
            print(f"- {contradiction}")
    else:
        print("No contradictions detected.")

    print("\nChecking internal contradictions for Newtonian Theory:")
    internal_newton = detector.detect_contradictions([theory_newton], context, types_to_detect=[ContradictionType.INTERNAL])
    if internal_newton:
        for contradiction in internal_newton:
            print(f"- {contradiction}")
    else:
        print("No internal contradictions detected in Newtonian Theory.")

    print("\nChecking internal contradictions for Einstein's Theory:")
    internal_einstein = detector.detect_contradictions([theory_einstein], context, types_to_detect=[ContradictionType.INTERNAL])
    if internal_einstein:
        for contradiction in internal_einstein:
            print(f"- {contradiction}")
    else:
        print("No internal contradictions detected in Einstein's Theory.")

