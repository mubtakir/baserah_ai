#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام تفسير الأحلام المتقدم بناءً على نظرية "العقل النعسان" لباسل يحيى عبدالله

هذا النظام يطبق المفاهيم المبتكرة من كتاب "العقل النعسان في فن تعبير الرؤى"
ويدمجها مع تقنيات الذكاء الاصطناعي الحديثة في نظام بصيرة.

المبادئ الأساسية:
1. العقل النعسان له قدرات متضاعفة أثناء النوم
2. الرؤيا تأتي رمزية وليست حرفية دائماً
3. أهمية السياق الشخصي للرائي (اسم، عمل، حالة، بيئة، لغة)
4. آليات الترميز المختلفة (تصحيف، جناس، إخراج فائق، قلب، استعارة، إلخ)
5. أنواع الأحلام المختلفة وما يُعبر منها وما لا يُعبر
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import re
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# استيراد مكونات بصيرة
try:
    from ..symbolic_processing.letter_semantics.semantic_analyzer import LetterSemanticAnalyzer
    from ..core.ai_oop.base_thing import Thing
    # إزالة التعلم المعزز التقليدي - لا نحتاجه في هذا النظام
    TRADITIONAL_RL_REMOVED = True
except ImportError:
    # في حالة التشغيل المباشر
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from symbolic_processing.letter_semantics.semantic_analyzer import LetterSemanticAnalyzer
        from core.ai_oop.base_thing import Thing
        # إزالة التعلم المعزز التقليدي
        TRADITIONAL_RL_REMOVED = True
    except ImportError:
        # إنشاء كلاسات وهمية للاختبار
        class LetterSemanticAnalyzer:
            def analyze_text(self, text): return {"semantic_analysis": "تحليل وهمي"}
        class Thing:
            def __init__(self, **kwargs): pass
        TRADITIONAL_RL_REMOVED = True

class DreamType(Enum):
    """أنواع الأحلام حسب تصنيف باسل"""
    TRUE_VISION = "رؤيا_صادقة"           # من الله
    SELF_TALK = "حديث_النفس"            # ما يشغل الإنسان
    TEMPERAMENT_REFLECTION = "انعكاس_طبع"  # تأثير المزاج والمرض
    SATAN_INTERFERENCE = "تلاعب_شيطان"   # تدخل شيطاني
    MIXED = "مختلط"                     # خليط من الأنواع

class SymbolicMechanism(Enum):
    """آليات الترميز في العقل النعسان"""
    TASHIF = "تصحيف"                    # قلب وتحريف الكلمات
    JINAS = "جناس"                      # كلمات متشابهة لفظاً مختلفة معنى
    EXAGGERATION = "إخراج_فائق"         # المبالغة والتضخيم/التحقير
    REVERSAL = "قلب_وعكس"              # البائع يشتري، الزبائن كلصوص
    METAPHOR = "استعارة_وتمثيل"         # استخدام شيء لتمثيل آخر
    RESULT_INDICATION = "إشارة_بالنتيجة" # الإشارة بالنتيجة عن المقدمة
    ABBREVIATION = "اختزال_وإكمال"      # اختزال المعروف وإكمال الناقص
    GENERALIZATION = "تعميم_وقياس"      # ترتيب الأمور في خانات متشابهة
    FAMOUS_REFERENCE = "اعتماد_مشهور"   # استخدام أسماء مشهورة للترميز
    PART_WHOLE = "جزء_كل"              # الترميز بالجزء عن الكل أو العكس
    OPPOSITES = "أضداد"                # استخدام الأضداد للترميز

@dataclass
class DreamerProfile:
    """ملف شخصي للرائي"""
    name: str
    nickname: Optional[str] = None
    profession: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    religion: Optional[str] = None
    temperament: Optional[str] = None  # دموي، صفراوي، سوداوي، بلغمي
    interests: List[str] = None
    current_concerns: List[str] = None
    health_status: Optional[str] = None
    social_status: Optional[str] = None
    language_dialect: Optional[str] = None
    cultural_background: Optional[str] = None

    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.current_concerns is None:
            self.current_concerns = []

@dataclass
class DreamElement:
    """عنصر من عناصر الحلم"""
    element: str
    category: str  # نبات، حيوان، جماد، شخص، مكان، حدث
    properties: Dict[str, Any]
    symbolic_meanings: List[str]
    personal_associations: List[str]
    linguistic_analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element": self.element,
            "category": self.category,
            "properties": self.properties,
            "symbolic_meanings": self.symbolic_meanings,
            "personal_associations": self.personal_associations,
            "linguistic_analysis": self.linguistic_analysis
        }

@dataclass
class BasilDreamInterpretation:
    """تفسير الحلم وفق نظرية باسل"""
    dream_text: str
    dreamer_profile: DreamerProfile
    dream_type: DreamType
    confidence_level: float
    elements: List[DreamElement]
    symbolic_mechanisms: List[SymbolicMechanism]
    interpretation_layers: Dict[str, str]
    overall_message: str
    recommendations: List[str]
    warnings: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_text": self.dream_text,
            "dreamer_profile": self.dreamer_profile.__dict__,
            "dream_type": self.dream_type.value,
            "confidence_level": self.confidence_level,
            "elements": [elem.to_dict() for elem in self.elements],
            "symbolic_mechanisms": [mech.value for mech in self.symbolic_mechanisms],
            "interpretation_layers": self.interpretation_layers,
            "overall_message": self.overall_message,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat()
        }

class BasilDreamInterpreter:
    """
    مفسر الأحلام وفق نظرية باسل يحيى عبدالله
    يطبق مفاهيم "العقل النعسان" وآليات الترميز المختلفة
    """

    def __init__(self, semantic_analyzer: LetterSemanticAnalyzer = None):
        self.semantic_analyzer = semantic_analyzer
        self.interpretation_history = []
        self.logger = logging.getLogger("basil_dream_interpreter")

        # تهيئة قواعد التصحيف والجناس
        self._init_linguistic_rules()

        # تهيئة قاعدة بيانات الرموز التراثية
        self._init_traditional_symbols()

        # تهيئة المحلل الدلالي إذا لم يتم توفيره
        if self.semantic_analyzer is None:
            try:
                from ..symbolic_processing.data.initial_letter_semantics_data2 import get_initial_letter_semantics_data
                semantics_data = get_initial_letter_semantics_data()
                self.semantic_analyzer = LetterSemanticAnalyzer(semantics_data)
            except ImportError:
                # إنشاء محلل دلالي وهمي للاختبار
                self.semantic_analyzer = LetterSemanticAnalyzer()

    def _init_linguistic_rules(self):
        """تهيئة قواعد التصحيف والجناس"""
        self.tashif_rules = {
            # قواعد التصحيف (قلب الحروف)
            "letter_swaps": {
                "ملح": "لحم",
                "دمشق": "دم_شق",
                "هدم_موصل": "مهد_وصل",
                "لوز": "زول"
            },
            # حذف النقاط
            "dot_removal": {
                "بسة": "بس",  # قطة -> فقط/رجاء
                "نبي": "ني",
                "تين": "تن"
            }
        }

        self.jinas_rules = {
            # الجناس (كلمات متشابهة لفظاً)
            "يحيى": ["يحيا"],
            "سنة": ["سِنة", "سُنة"],
            "ضرب": ["ضرب_بمعنى_السفر", "ضرب_بمعنى_الضرب"],
            "سوري": ["سور", "محراب"],
            "عزة": ["أعزاز"],
            "سعاد": ["السعودية"],
            "أمينة": ["اليمن"]
        }

    def _init_traditional_symbols(self):
        """تهيئة قاعدة بيانات الرموز التراثية"""
        self.traditional_symbols = {
            # رموز من التراث الإسلامي
            "ماء": {
                "meanings": ["حياة", "طهارة", "رزق", "علم", "رحمة"],
                "context_dependent": True
            },
            "نار": {
                "meanings": ["فتنة", "حرب", "سلطان", "غضب", "تطهير"],
                "context_dependent": True
            },
            "شمس": {
                "meanings": ["ملك", "سلطان", "عدل", "هداية", "والد"],
                "context_dependent": True
            },
            "قمر": {
                "meanings": ["وزير", "عالم", "جمال", "والدة", "زوجة"],
                "context_dependent": True
            },
            "شجرة": {
                "meanings": ["عمر", "نسل", "خير", "بركة", "رجل"],
                "context_dependent": True
            },
            "بيت": {
                "meanings": ["أمان", "عائلة", "استقرار", "زوجة", "جسد"],
                "context_dependent": True
            },
            "طيران": {
                "meanings": ["سفر", "علو_مقام", "تحرر", "موت", "روحانية"],
                "context_dependent": True
            },
            "موت": {
                "meanings": ["نهاية_مرحلة", "تغيير", "توبة", "زواج", "سفر"],
                "context_dependent": True
            },
            "حمامة_بيضاء": {
                "meanings": ["امرأة_شريفة", "رسالة_خير", "سلام"],
                "context_dependent": True
            },
            "صقر": {
                "meanings": ["رجل_قوي", "حاكم", "عربي_أصيل"],
                "context_dependent": True
            }
        }

    def interpret_dream(self, dream_text: str, dreamer_profile: DreamerProfile,
                       context: Optional[Dict[str, Any]] = None) -> BasilDreamInterpretation:
        """
        تفسير الحلم وفق نظرية باسل

        Args:
            dream_text: نص الحلم بالألفاظ الأصلية للرائي
            dreamer_profile: الملف الشخصي للرائي
            context: سياق إضافي

        Returns:
            تفسير شامل للحلم
        """
        self.logger.info(f"بدء تفسير حلم للرائي: {dreamer_profile.name}")

        context = context or {}

        # تحديد نوع الحلم
        dream_type = self._classify_dream_type(dream_text, dreamer_profile, context)

        # إذا كان الحلم من النوع الذي لا يُعبر، نتوقف هنا
        if dream_type in [DreamType.TEMPERAMENT_REFLECTION, DreamType.SATAN_INTERFERENCE]:
            return self._create_non_interpretable_result(dream_text, dreamer_profile, dream_type)

        # استخراج عناصر الحلم
        elements = self._extract_dream_elements(dream_text, dreamer_profile)

        # تحليل الآليات الرمزية المستخدمة
        symbolic_mechanisms = self._identify_symbolic_mechanisms(dream_text, elements, dreamer_profile)

        # تطبيق طبقات التفسير المختلفة
        interpretation_layers = self._apply_interpretation_layers(elements, dreamer_profile, symbolic_mechanisms)

        # توليد الرسالة الإجمالية
        overall_message = self._generate_overall_message(interpretation_layers, elements, dreamer_profile)

        # حساب مستوى الثقة
        confidence_level = self._calculate_confidence_level(dream_type, elements, dreamer_profile)

        # توليد التوصيات والتحذيرات
        recommendations = self._generate_recommendations(interpretation_layers, dreamer_profile)
        warnings = self._generate_warnings(dream_type, interpretation_layers)

        # إنشاء نتيجة التفسير
        interpretation = BasilDreamInterpretation(
            dream_text=dream_text,
            dreamer_profile=dreamer_profile,
            dream_type=dream_type,
            confidence_level=confidence_level,
            elements=elements,
            symbolic_mechanisms=symbolic_mechanisms,
            interpretation_layers=interpretation_layers,
            overall_message=overall_message,
            recommendations=recommendations,
            warnings=warnings,
            timestamp=datetime.now()
        )

        # حفظ التفسير في التاريخ
        self.interpretation_history.append(interpretation)

        self.logger.info(f"تم تفسير الحلم بنجاح، مستوى الثقة: {confidence_level:.2f}")

        return interpretation

    def _classify_dream_type(self, dream_text: str, dreamer_profile: DreamerProfile,
                           context: Dict[str, Any]) -> DreamType:
        """تصنيف نوع الحلم"""
        # فحص علامات الرؤيا الصادقة
        if self._has_true_vision_indicators(dream_text, dreamer_profile):
            return DreamType.TRUE_VISION

        # فحص علامات حديث النفس
        if self._has_self_talk_indicators(dream_text, dreamer_profile):
            return DreamType.SELF_TALK

        # فحص علامات انعكاس الطبع
        if self._has_temperament_indicators(dream_text, dreamer_profile):
            return DreamType.TEMPERAMENT_REFLECTION

        # فحص علامات التلاعب الشيطاني
        if self._has_satan_interference_indicators(dream_text):
            return DreamType.SATAN_INTERFERENCE

        # افتراضي: رؤيا صادقة مع احتمال الخلط
        return DreamType.TRUE_VISION

    def _has_true_vision_indicators(self, dream_text: str, dreamer_profile: DreamerProfile) -> bool:
        """فحص علامات الرؤيا الصادقة"""
        indicators = 0

        # الوضوح والقصر
        if len(dream_text.split()) < 50:  # حلم قصير
            indicators += 1

        # وجود رموز دينية أو روحانية
        religious_symbols = ["مسجد", "قرآن", "نبي", "ملك", "جنة", "نار", "صلاة"]
        if any(symbol in dream_text for symbol in religious_symbols):
            indicators += 2

        # الرائي صادق ومتدين
        if dreamer_profile.religion == "إسلام":
            indicators += 1

        return indicators >= 2

    def _has_self_talk_indicators(self, dream_text: str, dreamer_profile: DreamerProfile) -> bool:
        """فحص علامات حديث النفس"""
        # وجود اهتمامات الرائي الحالية في الحلم
        if dreamer_profile.current_concerns:
            for concern in dreamer_profile.current_concerns:
                if concern.lower() in dream_text.lower():
                    return True

        # وجود عمل الرائي في الحلم
        if dreamer_profile.profession and dreamer_profile.profession.lower() in dream_text.lower():
            return True

        return False

    def _has_temperament_indicators(self, dream_text: str, dreamer_profile: DreamerProfile) -> bool:
        """فحص علامات انعكاس الطبع والمرض"""
        if dreamer_profile.health_status == "مريض":
            return True

        # فحص علامات غلبة الأخلاط
        if dreamer_profile.temperament:
            if dreamer_profile.temperament == "دموي" and "أحمر" in dream_text:
                return True
            elif dreamer_profile.temperament == "صفراوي" and ("أصفر" in dream_text or "نار" in dream_text):
                return True
            elif dreamer_profile.temperament == "سوداوي" and ("أسود" in dream_text or "دخان" in dream_text):
                return True
            elif dreamer_profile.temperament == "بلغمي" and ("أبيض" in dream_text or "ماء" in dream_text):
                return True

        return False

    def _has_satan_interference_indicators(self, dream_text: str) -> bool:
        """فحص علامات التلاعب الشيطاني"""
        negative_indicators = ["خوف", "رعب", "كابوس", "شيطان", "وحش", "ظلام"]
        return any(indicator in dream_text for indicator in negative_indicators)

    def _extract_dream_elements(self, dream_text: str, dreamer_profile: DreamerProfile) -> List[DreamElement]:
        """استخراج عناصر الحلم وتحليلها"""
        elements = []

        # تنظيف النص وتقسيمه
        words = re.findall(r'\b\w+\b', dream_text)

        for word in words:
            # تحليل دلالي للكلمة
            try:
                semantic_analysis = self.semantic_analyzer.analyze_text(word)
            except:
                semantic_analysis = {"semantic_analysis": "تحليل أساسي"}

            # تحديد فئة العنصر
            category = self._categorize_element(word)

            # استخراج المعاني الرمزية
            symbolic_meanings = self._get_symbolic_meanings(word, dreamer_profile)

            # الربط الشخصي
            personal_associations = self._get_personal_associations(word, dreamer_profile)

            if symbolic_meanings or personal_associations:
                element = DreamElement(
                    element=word,
                    category=category,
                    properties={},
                    symbolic_meanings=symbolic_meanings,
                    personal_associations=personal_associations,
                    linguistic_analysis=semantic_analysis or {}
                )
                elements.append(element)

        return elements

    def _categorize_element(self, word: str) -> str:
        """تصنيف عنصر الحلم"""
        # قواعد بسيطة للتصنيف - يمكن تطويرها
        animals = ["أسد", "ذئب", "كلب", "قطة", "بسة", "حوت", "سمك", "طائر", "حمامة", "صقر"]
        plants = ["شجرة", "نخلة", "ورد", "زهرة", "عشب"]
        places = ["بيت", "مسجد", "مدرسة", "سوق", "جبل", "بحر", "نهر"]
        people = ["رجل", "امرأة", "طفل", "شيخ", "نبي", "ملك"]
        objects = ["كتاب", "سيارة", "طائرة", "مفتاح", "باب", "نافذة"]

        if word in animals:
            return "حيوان"
        elif word in plants:
            return "نبات"
        elif word in places:
            return "مكان"
        elif word in people:
            return "شخص"
        elif word in objects:
            return "جماد"
        else:
            return "غير_محدد"

    def _get_symbolic_meanings(self, word: str, dreamer_profile: DreamerProfile) -> List[str]:
        """استخراج المعاني الرمزية للكلمة"""
        meanings = []

        # البحث في قاعدة الرموز التراثية
        if word in self.traditional_symbols:
            meanings.extend(self.traditional_symbols[word]["meanings"])

        # تطبيق قواعد التصحيف
        tashif_meanings = self._apply_tashif_rules(word)
        meanings.extend(tashif_meanings)

        # تطبيق قواعد الجناس
        jinas_meanings = self._apply_jinas_rules(word, dreamer_profile)
        meanings.extend(jinas_meanings)

        return list(set(meanings))  # إزالة التكرار

    def _apply_tashif_rules(self, word: str) -> List[str]:
        """تطبيق قواعد التصحيف"""
        meanings = []

        # البحث في قواعد التصحيف المحددة مسبقاً
        if word in self.tashif_rules["letter_swaps"]:
            meanings.append(self.tashif_rules["letter_swaps"][word])

        # تطبيق قواعد حذف النقاط
        if word in self.tashif_rules["dot_removal"]:
            meanings.append(self.tashif_rules["dot_removal"][word])

        return meanings

    def _apply_jinas_rules(self, word: str, dreamer_profile: DreamerProfile) -> List[str]:
        """تطبيق قواعد الجناس"""
        meanings = []

        # البحث في قواعد الجناس
        if word in self.jinas_rules:
            meanings.extend(self.jinas_rules[word])

        # تطبيق قواعد خاصة بالرائي (مثل الأسماء الجغرافية)
        if dreamer_profile.cultural_background == "عربي":
            if word == "سوري":
                meanings.append("اقتحام_السور")
            elif word == "عزة":
                meanings.append("مدينة_أعزاز")

        return meanings

    def _get_personal_associations(self, word: str, dreamer_profile: DreamerProfile) -> List[str]:
        """الحصول على الربط الشخصي للكلمة"""
        associations = []

        # ربط بالاسم
        if dreamer_profile.name and word in dreamer_profile.name:
            associations.append(f"مرتبط_بالاسم_{dreamer_profile.name}")

        # ربط بالمهنة
        if dreamer_profile.profession and word.lower() in dreamer_profile.profession.lower():
            associations.append(f"مرتبط_بالمهنة_{dreamer_profile.profession}")

        # ربط بالاهتمامات
        for interest in dreamer_profile.interests:
            if word.lower() in interest.lower():
                associations.append(f"مرتبط_بالاهتمام_{interest}")

        return associations

    def _identify_symbolic_mechanisms(self, dream_text: str, elements: List[DreamElement],
                                    dreamer_profile: DreamerProfile) -> List[SymbolicMechanism]:
        """تحديد الآليات الرمزية المستخدمة في الحلم"""
        mechanisms = []

        # فحص التصحيف
        if any("تصحيف" in elem.symbolic_meanings for elem in elements):
            mechanisms.append(SymbolicMechanism.TASHIF)

        # فحص الجناس
        if any("جناس" in str(elem.symbolic_meanings) for elem in elements):
            mechanisms.append(SymbolicMechanism.JINAS)

        # فحص المبالغة (كلمات تدل على الكبر أو الصغر المفرط)
        exaggeration_words = ["عملاق", "ضخم", "صغير_جداً", "كبير_جداً"]
        if any(word in dream_text for word in exaggeration_words):
            mechanisms.append(SymbolicMechanism.EXAGGERATION)

        # فحص القلب والعكس
        reversal_indicators = ["البائع_يشتري", "الطفل_كبير", "الميت_حي"]
        if any(indicator in dream_text for indicator in reversal_indicators):
            mechanisms.append(SymbolicMechanism.REVERSAL)

        return mechanisms

    def _apply_interpretation_layers(self, elements: List[DreamElement],
                                   dreamer_profile: DreamerProfile,
                                   mechanisms: List[SymbolicMechanism]) -> Dict[str, str]:
        """تطبيق طبقات التفسير المختلفة"""
        layers = {}

        # الطبقة الحرفية
        layers["حرفي"] = self._literal_interpretation(elements)

        # الطبقة الرمزية
        layers["رمزي"] = self._symbolic_interpretation(elements, mechanisms)

        # الطبقة الشخصية
        layers["شخصي"] = self._personal_interpretation(elements, dreamer_profile)

        # الطبقة الدينية/الروحية
        layers["ديني_روحي"] = self._religious_interpretation(elements, dreamer_profile)

        # الطبقة النفسية
        layers["نفسي"] = self._psychological_interpretation(elements, dreamer_profile)

        return layers

    def _literal_interpretation(self, elements: List[DreamElement]) -> str:
        """التفسير الحرفي"""
        if not elements:
            return "لا توجد عناصر واضحة للتفسير الحرفي."

        interpretation = "التفسير الحرفي: "
        for element in elements[:3]:  # أخذ أول 3 عناصر
            if element.symbolic_meanings:
                interpretation += f"{element.element} قد يشير إلى {element.symbolic_meanings[0]}. "

        return interpretation

    def _symbolic_interpretation(self, elements: List[DreamElement],
                               mechanisms: List[SymbolicMechanism]) -> str:
        """التفسير الرمزي"""
        interpretation = "التفسير الرمزي: "

        # تحليل الآليات المستخدمة
        if SymbolicMechanism.TASHIF in mechanisms:
            interpretation += "يستخدم الحلم آلية التصحيف (قلب الحروف) للترميز. "

        if SymbolicMechanism.JINAS in mechanisms:
            interpretation += "يستخدم الحلم آلية الجناس (التشابه اللفظي) للترميز. "

        # تحليل العلاقات بين العناصر
        if len(elements) > 1:
            interpretation += f"التفاعل بين {elements[0].element} و{elements[1].element} يشير إلى تغيير أو تحول في حياة الرائي. "

        return interpretation

    def _personal_interpretation(self, elements: List[DreamElement],
                               dreamer_profile: DreamerProfile) -> str:
        """التفسير الشخصي"""
        interpretation = "التفسير الشخصي: "

        # ربط بالمهنة
        if dreamer_profile.profession:
            profession_related = [elem for elem in elements if any("مهنة" in assoc for assoc in elem.personal_associations)]
            if profession_related:
                interpretation += f"الحلم يرتبط بمهنة الرائي ({dreamer_profile.profession}). "

        # ربط بالاهتمامات
        if dreamer_profile.interests:
            interest_related = [elem for elem in elements if any("اهتمام" in assoc for assoc in elem.personal_associations)]
            if interest_related:
                interpretation += "الحلم يعكس اهتمامات الرائي الحالية. "

        # ربط بالهموم الحالية
        if dreamer_profile.current_concerns:
            interpretation += "الحلم قد يكون انعكاساً لهموم الرائي الحالية. "

        return interpretation

    def _religious_interpretation(self, elements: List[DreamElement],
                                dreamer_profile: DreamerProfile) -> str:
        """التفسير الديني/الروحي"""
        interpretation = "التفسير الديني والروحي: "

        # البحث عن رموز دينية
        religious_elements = [elem for elem in elements if any(
            meaning in ["رحمة", "هداية", "بركة", "تطهير", "توبة"]
            for meaning in elem.symbolic_meanings
        )]

        if religious_elements:
            interpretation += "الحلم يحمل رسالة روحية إيجابية. "

        # تحليل حسب دين الرائي
        if dreamer_profile.religion == "إسلام":
            interpretation += "من منظور إسلامي، هذا الحلم قد يكون بشارة أو تنبيه من الله. "

        return interpretation

    def _psychological_interpretation(self, elements: List[DreamElement],
                                    dreamer_profile: DreamerProfile) -> str:
        """التفسير النفسي"""
        interpretation = "التفسير النفسي: "

        # تحليل الحالة النفسية من خلال العناصر
        positive_elements = [elem for elem in elements if any(
            meaning in ["فرح", "سعادة", "نجاح", "خير"]
            for meaning in elem.symbolic_meanings
        )]

        negative_elements = [elem for elem in elements if any(
            meaning in ["خوف", "قلق", "حزن", "فشل"]
            for meaning in elem.symbolic_meanings
        )]

        if positive_elements:
            interpretation += "الحلم يعكس حالة نفسية إيجابية أو تطلعات مستقبلية. "

        if negative_elements:
            interpretation += "الحلم قد يعكس قلق أو توتر نفسي يحتاج للانتباه. "

        return interpretation

    def _generate_overall_message(self, interpretation_layers: Dict[str, str],
                                elements: List[DreamElement],
                                dreamer_profile: DreamerProfile) -> str:
        """توليد الرسالة الإجمالية للحلم"""
        message = "الرسالة الإجمالية للحلم:\n\n"

        # دمج التفسيرات من الطبقات المختلفة
        for layer_name, layer_interpretation in interpretation_layers.items():
            message += f"• {layer_name}: {layer_interpretation}\n"

        message += "\nالخلاصة: "

        # تحديد الاتجاه العام للحلم
        positive_indicators = sum(1 for elem in elements if any(
            meaning in ["خير", "بركة", "نجاح", "فرح"]
            for meaning in elem.symbolic_meanings
        ))

        negative_indicators = sum(1 for elem in elements if any(
            meaning in ["شر", "خوف", "فشل", "حزن"]
            for meaning in elem.symbolic_meanings
        ))

        if positive_indicators > negative_indicators:
            message += "هذا الحلم يحمل بشارة خير وإيجابية للرائي. "
        elif negative_indicators > positive_indicators:
            message += "هذا الحلم يحمل تنبيهاً أو تحذيراً يستدعي الانتباه. "
        else:
            message += "هذا الحلم متوازن ويحمل رسائل متعددة الأبعاد. "

        return message

    def _calculate_confidence_level(self, dream_type: DreamType, elements: List[DreamElement],
                                  dreamer_profile: DreamerProfile) -> float:
        """حساب مستوى الثقة في التفسير"""
        confidence = 0.5  # قيمة أساسية

        # زيادة الثقة حسب نوع الحلم
        if dream_type == DreamType.TRUE_VISION:
            confidence += 0.3
        elif dream_type == DreamType.SELF_TALK:
            confidence += 0.1

        # زيادة الثقة حسب عدد العناصر المفهومة
        understood_elements = len([elem for elem in elements if elem.symbolic_meanings])
        confidence += min(0.2, understood_elements * 0.05)

        # زيادة الثقة إذا كان لدينا معلومات كافية عن الرائي
        if dreamer_profile.profession and dreamer_profile.interests:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_recommendations(self, interpretation_layers: Dict[str, str],
                                dreamer_profile: DreamerProfile) -> List[str]:
        """توليد التوصيات"""
        recommendations = [
            "تأمل في معاني الحلم وربطها بواقعك الحالي",
            "استشر أهل العلم إذا كان الحلم يحمل رسالة مهمة",
            "لا تبني قرارات مصيرية على تفسير الأحلام وحده"
        ]

        # توصيات خاصة بالرائي
        if dreamer_profile.religion == "إسلام":
            recommendations.append("أكثر من الاستغفار والذكر قبل النوم")
            recommendations.append("احرص على الوضوء قبل النوم لصفاء الرؤى")

        return recommendations

    def _generate_warnings(self, dream_type: DreamType, interpretation_layers: Dict[str, str]) -> List[str]:
        """توليد التحذيرات"""
        warnings = [
            "هذا التفسير اجتهادي وليس قطعياً",
            "الأحلام قد تكون أضغاث أحلام لا معنى لها",
            "لا تدع القلق من تفسير الحلم يؤثر على حياتك"
        ]

        # تحذيرات خاصة حسب نوع الحلم
        if dream_type == DreamType.TEMPERAMENT_REFLECTION:
            warnings.append("هذا الحلم قد يكون انعكاساً لحالة جسدية وليس رؤيا")

        if dream_type == DreamType.SATAN_INTERFERENCE:
            warnings.append("إذا كان الحلم مخيفاً، استعذ بالله ولا تحدث به أحداً")

        return warnings

    def _create_non_interpretable_result(self, dream_text: str, dreamer_profile: DreamerProfile,
                                       dream_type: DreamType) -> BasilDreamInterpretation:
        """إنشاء نتيجة للأحلام التي لا تُعبر"""
        message = ""
        recommendations = []
        warnings = []

        if dream_type == DreamType.TEMPERAMENT_REFLECTION:
            message = "هذا الحلم يبدو أنه انعكاس لحالة جسدية أو مزاجية وليس رؤيا تُعبر."
            recommendations = ["راجع حالتك الصحية", "احرص على نظام غذائي متوازن"]
            warnings = ["هذا النوع من الأحلام لا يُعبر عادة"]

        elif dream_type == DreamType.SATAN_INTERFERENCE:
            message = "هذا الحلم يحمل علامات التلاعب الشيطاني ولا يُعبر."
            recommendations = ["استعذ بالله من الشيطان", "أكثر من الذكر والاستغفار"]
            warnings = ["لا تحدث بهذا الحلم أحداً", "لا تدع الخوف يسيطر عليك"]

        return BasilDreamInterpretation(
            dream_text=dream_text,
            dreamer_profile=dreamer_profile,
            dream_type=dream_type,
            confidence_level=0.9,  # ثقة عالية في عدم التعبير
            elements=[],
            symbolic_mechanisms=[],
            interpretation_layers={"تصنيف": message},
            overall_message=message,
            recommendations=recommendations,
            warnings=warnings,
            timestamp=datetime.now()
        )

    def get_interpretation_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التفسيرات"""
        if not self.interpretation_history:
            return {"total_interpretations": 0}

        total = len(self.interpretation_history)
        dream_types = {}
        avg_confidence = sum(interp.confidence_level for interp in self.interpretation_history) / total

        for interp in self.interpretation_history:
            dream_type = interp.dream_type.value
            dream_types[dream_type] = dream_types.get(dream_type, 0) + 1

        return {
            "total_interpretations": total,
            "average_confidence": avg_confidence,
            "dream_types_distribution": dream_types,
            "most_common_mechanisms": self._get_most_common_mechanisms()
        }

    def _get_most_common_mechanisms(self) -> Dict[str, int]:
        """الحصول على أكثر الآليات الرمزية استخداماً"""
        mechanism_counts = {}

        for interp in self.interpretation_history:
            for mechanism in interp.symbolic_mechanisms:
                mech_name = mechanism.value
                mechanism_counts[mech_name] = mechanism_counts.get(mech_name, 0) + 1

        return dict(sorted(mechanism_counts.items(), key=lambda x: x[1], reverse=True))

# دالة مساعدة لإنشاء مفسر الأحلام
def create_basil_dream_interpreter() -> BasilDreamInterpreter:
    """إنشاء مفسر أحلام وفق نظرية باسل"""
    return BasilDreamInterpreter()

# دالة مساعدة لإنشاء ملف شخصي للرائي
def create_dreamer_profile(name: str, **kwargs) -> DreamerProfile:
    """إنشاء ملف شخصي للرائي"""
    return DreamerProfile(name=name, **kwargs)
