#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك تفسير الرموز ودلالات الحروف (Symbolic Interpretation Engine)

هذه الوحدة تمثل "العقل الدلالي" لنظام بصيرة، وتوفر الآليات الأساسية لتفسير الرموز اللغوية،
بدءاً من الحروف الفردية ووصولاً إلى الكلمات والمفاهيم. تطبق هذه الوحدة رؤية باسل حول
"معاني الحروف" المستمدة من الصوت والشكل والسياقات الأخرى، وتوفر آليات "تجميع" أو "تفاعل"
دلالات الحروف لتكوين "بصمة دلالية" للكلمات.

المؤلف: نظام بصيرة (Basira System)
تاريخ الإنشاء: 20 مايو 2025
"""

from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum
import json
import os
import re
from dataclasses import dataclass, field, asdict
import copy
import math
from collections import defaultdict

# -----------------------------------------------------------------------------
# فئات المساعدة والتعدادات
# -----------------------------------------------------------------------------

class Language(str, Enum):
    """تعداد للغات المدعومة في النظام."""
    ARABIC = "ar"
    ENGLISH = "en"
    
    @classmethod
    def from_string(cls, lang_str: str) -> 'Language':
        """تحويل سلسلة نصية إلى نوع اللغة المقابل."""
        lang_str = lang_str.lower()
        if lang_str in ["ar", "arabic", "العربية"]:
            return cls.ARABIC
        elif lang_str in ["en", "english", "الإنجليزية"]:
            return cls.ENGLISH
        else:
            raise ValueError(f"اللغة غير مدعومة: {lang_str}")

class SemanticAxisPolarity(float, Enum):
    """تعداد لقطبية المحور الدلالي."""
    NEGATIVE = -1.0  # القطب السالب للمحور
    NEUTRAL = 0.0    # نقطة الحياد
    POSITIVE = 1.0   # القطب الموجب للمحور

class SemanticSynthesisMethod(str, Enum):
    """تعداد لطرق تجميع الدلالات."""
    SEQUENTIAL_NARRATIVE = "sequential_narrative"  # الآلية التسلسلية/القصصية
    OVERLAY_BLEND = "overlay_blend"                # آلية التراكب/المزيج
    SEMANTIC_AXES = "semantic_axes"                # آلية المحاور الدلالية
    DOMINANT_SYMBOL = "dominant_symbol"            # آلية "الرمز المهيمن"
    ALL = "all"                                    # استخدام جميع الطرق

@dataclass
class SemanticAxis:
    """تمثيل لمحور دلالي مع قطبيه."""
    name: str                      # اسم المحور (مثل "الحركة")
    negative_pole: str             # القطب السالب (مثل "سكون")
    positive_pole: str             # القطب الموجب (مثل "تدفق")
    description: Optional[str] = None  # وصف اختياري للمحور
    
    def get_pole_at_intensity(self, intensity: float) -> str:
        """الحصول على وصف للمعنى عند شدة معينة على المحور.
        
        المعاملات:
            intensity: قيمة بين -1 و 1 تمثل الشدة والاتجاه على المحور
            
        العائد:
            وصف نصي للمعنى عند هذه النقطة على المحور
        """
        if intensity < -0.7:
            return f"{self.negative_pole} بشكل قوي"
        elif intensity < -0.3:
            return f"{self.negative_pole} بشكل معتدل"
        elif intensity < 0.3:
            return f"متوازن بين {self.negative_pole} و{self.positive_pole}"
        elif intensity < 0.7:
            return f"{self.positive_pole} بشكل معتدل"
        else:
            return f"{self.positive_pole} بشكل قوي"

@dataclass
class PhoneticProperties:
    """خصائص صوتية للحرف."""
    articulation_point: str        # مكان النطق (مثل "شفوي", "حلقي")
    articulation_method: str       # طريقة النطق (مثل "انفجاري", "احتكاكي")
    is_vowel: bool = False         # هل هو حرف علة
    is_consonant: bool = True      # هل هو حرف ساكن
    is_emphatic: bool = False      # هل هو حرف مفخم
    natural_sounds: List[str] = field(default_factory=list)  # أصوات طبيعية مرتبطة
    
    def to_dict(self) -> Dict:
        """تحويل الكائن إلى قاموس."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhoneticProperties':
        """إنشاء كائن من قاموس."""
        return cls(**data)

# -----------------------------------------------------------------------------
# فئة LetterSemantics - تمثيل الدلالات المتعددة لحرف واحد
# -----------------------------------------------------------------------------

@dataclass
class LetterSemantics:
    """تمثيل الدلالات المتعددة لحرف واحد."""
    
    character: str                 # الحرف نفسه (مثلاً "أ", "ب", "A", "B")
    language: Language             # اللغة ("ar" أو "en")
    
    # الخصائص الصوتية
    phonetic_properties: PhoneticProperties
    
    # الدلالات المستمدة من شكل الحرف
    visual_form_semantics: List[str] = field(default_factory=list)
    
    # المحاور الدلالية الأساسية
    core_semantic_axes: Dict[str, SemanticAxis] = field(default_factory=dict)
    
    # دلالات عامة أخرى
    general_connotations: List[str] = field(default_factory=list)
    
    # الصدى العاطفي (مستقبلاً)
    emotional_resonance: Dict[str, float] = field(default_factory=dict)
    
    # معلومات إضافية
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """تنفيذ عمليات ما بعد التهيئة."""
        # تحويل اللغة إلى نوع Language إذا كانت سلسلة نصية
        if isinstance(self.language, str):
            self.language = Language.from_string(self.language)
        
        # تحويل PhoneticProperties إلى كائن إذا كان قاموساً
        if isinstance(self.phonetic_properties, dict):
            self.phonetic_properties = PhoneticProperties.from_dict(self.phonetic_properties)
        
        # تحويل المحاور الدلالية إلى كائنات SemanticAxis إذا كانت tuples
        axes_dict = {}
        for axis_name, axis_data in self.core_semantic_axes.items():
            if isinstance(axis_data, tuple) and len(axis_data) >= 2:
                # إذا كان axis_data هو tuple من شكل (negative_pole, positive_pole, [description])
                negative_pole, positive_pole = axis_data[0], axis_data[1]
                description = axis_data[2] if len(axis_data) > 2 else None
                axes_dict[axis_name] = SemanticAxis(
                    name=axis_name,
                    negative_pole=negative_pole,
                    positive_pole=positive_pole,
                    description=description
                )
            elif isinstance(axis_data, SemanticAxis):
                # إذا كان axis_data هو كائن SemanticAxis بالفعل
                axes_dict[axis_name] = axis_data
        
        self.core_semantic_axes = axes_dict
    
    def get_meaning_along_axis(self, axis_name: str, intensity: float) -> str:
        """الحصول على معنى محدد على محور دلالي بناءً على شدة معينة.
        
        المعاملات:
            axis_name: اسم المحور الدلالي
            intensity: قيمة بين -1 و 1 تمثل الشدة والاتجاه على المحور
            
        العائد:
            وصف نصي للمعنى عند هذه النقطة على المحور
            
        يرفع:
            KeyError: إذا كان المحور غير موجود
        """
        if axis_name not in self.core_semantic_axes:
            raise KeyError(f"المحور الدلالي '{axis_name}' غير موجود للحرف '{self.character}'")
        
        # تقييد intensity بين -1 و 1
        intensity = max(-1.0, min(1.0, intensity))
        
        return self.core_semantic_axes[axis_name].get_pole_at_intensity(intensity)
    
    def get_dominant_connotation(self) -> str:
        """الحصول على الدلالة المهيمنة للحرف.
        
        العائد:
            الدلالة الأكثر أهمية للحرف
        """
        if self.general_connotations:
            return self.general_connotations[0]
        elif self.visual_form_semantics:
            return self.visual_form_semantics[0]
        elif self.core_semantic_axes:
            # استخدام أول محور دلالي
            first_axis = next(iter(self.core_semantic_axes.values()))
            return f"محور {first_axis.name}: بين {first_axis.negative_pole} و{first_axis.positive_pole}"
        else:
            return "لا توجد دلالة مهيمنة"
    
    def to_dict(self) -> Dict:
        """تحويل الكائن إلى قاموس.
        
        العائد:
            قاموس يمثل الكائن
        """
        result = {
            "character": self.character,
            "language": self.language.value,
            "phonetic_properties": self.phonetic_properties.to_dict(),
            "visual_form_semantics": self.visual_form_semantics,
            "general_connotations": self.general_connotations,
            "emotional_resonance": self.emotional_resonance,
        }
        
        # تحويل المحاور الدلالية إلى قواميس
        result["core_semantic_axes"] = {
            axis_name: {
                "name": axis.name,
                "negative_pole": axis.negative_pole,
                "positive_pole": axis.positive_pole,
                "description": axis.description
            }
            for axis_name, axis in self.core_semantic_axes.items()
        }
        
        if self.description:
            result["description"] = self.description
        
        if self.examples:
            result["examples"] = self.examples
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LetterSemantics':
        """إنشاء كائن من قاموس.
        
        المعاملات:
            data: قاموس يحتوي على بيانات الحرف
            
        العائد:
            كائن LetterSemantics جديد
        """
        # نسخة من البيانات لتجنب تعديل الأصل
        data_copy = copy.deepcopy(data)
        
        # استخراج المحاور الدلالية وتحويلها إلى الشكل المطلوب
        core_semantic_axes = {}
        if "core_semantic_axes" in data_copy:
            axes_data = data_copy.pop("core_semantic_axes")
            for axis_name, axis_info in axes_data.items():
                if isinstance(axis_info, dict):
                    core_semantic_axes[axis_name] = SemanticAxis(
                        name=axis_info.get("name", axis_name),
                        negative_pole=axis_info["negative_pole"],
                        positive_pole=axis_info["positive_pole"],
                        description=axis_info.get("description")
                    )
                elif isinstance(axis_info, (list, tuple)) and len(axis_info) >= 2:
                    core_semantic_axes[axis_name] = SemanticAxis(
                        name=axis_name,
                        negative_pole=axis_info[0],
                        positive_pole=axis_info[1],
                        description=axis_info[2] if len(axis_info) > 2 else None
                    )
            
            data_copy["core_semantic_axes"] = core_semantic_axes
        
        # تحويل الخصائص الصوتية إلى كائن PhoneticProperties
        if "phonetic_properties" in data_copy and isinstance(data_copy["phonetic_properties"], dict):
            data_copy["phonetic_properties"] = PhoneticProperties.from_dict(data_copy["phonetic_properties"])
        
        return cls(**data_copy)

# -----------------------------------------------------------------------------
# فئة SemanticDatabase - تخزين وإدارة كائنات LetterSemantics
# -----------------------------------------------------------------------------

class SemanticDatabase:
    """تخزين وإدارة كائنات LetterSemantics لجميع الحروف المدعومة في اللغات المختلفة."""
    
    def __init__(self, data_file: Optional[str] = None):
        """تهيئة قاعدة البيانات الدلالية.
        
        المعاملات:
            data_file: مسار ملف JSON يحتوي على بيانات الحروف (اختياري)
        """
        # قاموس لتخزين دلالات الحروف، مفتاح بواسطة اللغة ثم الحرف
        self.db: Dict[str, Dict[str, LetterSemantics]] = {
            Language.ARABIC.value: {},
            Language.ENGLISH.value: {}
        }
        
        # تحميل البيانات من ملف إذا تم تحديده
        if data_file and os.path.exists(data_file):
            self.load_from_file(data_file)
        else:
            # تهيئة قاعدة البيانات بالقيم الافتراضية
            self._initialize_default_database()
    
    def _initialize_default_database(self):
        """تهيئة قاعدة البيانات بالقيم الافتراضية بناءً على التفسيرات الرمزية للحروف."""
        # تهيئة الحروف العربية
        self._initialize_arabic_letters()
        
        # تهيئة الحروف الإنجليزية
        self._initialize_english_letters()
    
    def _initialize_arabic_letters(self):
        """تهيئة دلالات الحروف العربية."""
        # الألف (ا)
        self.add_letter_semantics(LetterSemantics(
            character="ا",
            language=Language.ARABIC,
            phonetic_properties=PhoneticProperties(
                articulation_point="حنجري",
                articulation_method="مد",
                is_vowel=True,
                is_consonant=False,
                natural_sounds=["صوت التنفس", "صوت الهواء"]
            ),
            visual_form_semantics=["استقامة", "ارتفاع", "عمود", "قوة"],
            core_semantic_axes={
                "الارتفاع": SemanticAxis(
                    name="الارتفاع",
                    negative_pole="انخفاض",
                    positive_pole="علو",
                    description="محور يمثل الارتفاع المادي أو المعنوي"
                ),
                "الاستقامة": SemanticAxis(
                    name="الاستقامة",
                    negative_pole="انحناء",
                    positive_pole="استقامة",
                    description="محور يمثل الاستقامة الشكلية أو المعنوية"
                )
            },
            general_connotations=["البداية", "الأصل", "العظمة", "الوحدانية", "الأولوية"],
            emotional_resonance={"قوة": 0.8, "ثبات": 0.7, "وضوح": 0.9},
            description="الألف هو أول الحروف العربية، ويرمز للبداية والأصل والعظمة",
            examples=["أمر", "أول", "أحد"]
        ))
        
        # الباء (ب)
        self.add_letter_semantics(LetterSemantics(
            character="ب",
            language=Language.ARABIC,
            phonetic_properties=PhoneticProperties(
                articulation_point="شفوي",
                articulation_method="انفجاري",
                natural_sounds=["صوت الانفجار الخفيف", "صوت الماء"]
            ),
            visual_form_semantics=["وعاء", "احتواء", "تجويف"],
            core_semantic_axes={
                "الاحتواء": SemanticAxis(
                    name="الاحتواء",
                    negative_pole="تفريغ",
                    positive_pole="امتلاء",
                    description="محور يمثل الاحتواء والامتلاء"
                ),
                "الظهور": SemanticAxis(
                    name="الظهور",
                    negative_pole="خفاء",
                    positive_pole="بروز",
                    description="محور يمثل الظهور والبروز"
                )
            },
            general_connotations=["الاحتواء", "البيت", "البناء", "البركة", "البداية الفعلية"],
            emotional_resonance={"أمان": 0.6, "استقرار": 0.7, "خصوبة": 0.8},
            description="الباء يرمز للاحتواء والبناء والبداية الفعلية للأشياء",
            examples=["بيت", "بناء", "بحر"]
        ))
        
        # الجيم (ج)
        self.add_letter_semantics(LetterSemantics(
            character="ج",
            language=Language.ARABIC,
            phonetic_properties=PhoneticProperties(
                articulation_point="وسط الحنك",
                articulation_method="مركب",
                natural_sounds=["صوت التجمع", "صوت الحركة المفاجئة"]
            ),
            visual_form_semantics=["تجويف", "تجمع", "انحناء"],
            core_semantic_axes={
                "التجمع": SemanticAxis(
                    name="التجمع",
                    negative_pole="تفرق",
                    positive_pole="تجمع",
                    description="محور يمثل التجمع والتفرق"
                ),
                "الجمال": SemanticAxis(
                    name="الجمال",
                    negative_pole="قبح",
                    positive_pole="جمال",
                    description="محور يمثل الجمال والقبح"
                )
            },
            general_connotations=["التجمع", "الجمال", "الجوهر", "الجسم", "الجهد"],
            emotional_resonance={"انسجام": 0.7, "تناغم": 0.6, "حيوية": 0.8},
            description="الجيم يرمز للتجمع والجمال والجوهر",
            examples=["جمال", "جمع", "جبل"]
        ))
        
        # إضافة المزيد من الحروف العربية هنا...
    
    def _initialize_english_letters(self):
        """تهيئة دلالات الحروف الإنجليزية."""
        # الحرف A
        self.add_letter_semantics(LetterSemantics(
            character="A",
            language=Language.ENGLISH,
            phonetic_properties=PhoneticProperties(
                articulation_point="مفتوح",
                articulation_method="مد",
                is_vowel=True,
                is_consonant=False,
                natural_sounds=["صوت الانفتاح", "صوت الإعلان"]
            ),
            visual_form_semantics=["هرم", "قمة", "سقف", "جبل"],
            core_semantic_axes={
                "الارتفاع": SemanticAxis(
                    name="الارتفاع",
                    negative_pole="انخفاض",
                    positive_pole="علو",
                    description="محور يمثل الارتفاع المادي أو المعنوي"
                ),
                "الانفتاح": SemanticAxis(
                    name="الانفتاح",
                    negative_pole="انغلاق",
                    positive_pole="انفتاح",
                    description="محور يمثل الانفتاح والانغلاق"
                )
            },
            general_connotations=["البداية", "الأول", "الإنجاز", "التفوق", "الأصالة"],
            emotional_resonance={"قوة": 0.8, "ثقة": 0.7, "طموح": 0.9},
            description="الحرف A يرمز للبداية والتفوق والأصالة",
            examples=["Achievement", "Apex", "Authority"]
        ))
        
        # الحرف B
        self.add_letter_semantics(LetterSemantics(
            character="B",
            language=Language.ENGLISH,
            phonetic_properties=PhoneticProperties(
                articulation_point="شفوي",
                articulation_method="انفجاري",
                natural_sounds=["صوت الانفجار الخفيف", "صوت الفقاعة"]
            ),
            visual_form_semantics=["بطن", "حمل", "احتواء", "ازدواجية"],
            core_semantic_axes={
                "الاحتواء": SemanticAxis(
                    name="الاحتواء",
                    negative_pole="تفريغ",
                    positive_pole="امتلاء",
                    description="محور يمثل الاحتواء والامتلاء"
                ),
                "الازدواجية": SemanticAxis(
                    name="الازدواجية",
                    negative_pole="أحادية",
                    positive_pole="ثنائية",
                    description="محور يمثل الأحادية والثنائية"
                )
            },
            general_connotations=["البناء", "البداية", "الازدواجية", "الاحتواء", "الولادة"],
            emotional_resonance={"أمان": 0.6, "استقرار": 0.7, "خصوبة": 0.8},
            description="الحرف B يرمز للبناء والاحتواء والازدواجية",
            examples=["Build", "Birth", "Balance"]
        ))
        
        # إضافة المزيد من الحروف الإنجليزية هنا...
    
    def get_letter_semantics(self, char: str, lang: Union[str, Language]) -> Optional[LetterSemantics]:
        """استرجاع دلالات حرف معين.
        
        المعاملات:
            char: الحرف المراد استرجاع دلالاته
            lang: اللغة ("ar" أو "en" أو كائن Language)
            
        العائد:
            كائن LetterSemantics أو None إذا لم يتم العثور على الحرف
        """
        # تحويل اللغة إلى سلسلة نصية إذا كانت كائن Language
        if isinstance(lang, Language):
            lang = lang.value
        
        # التأكد من أن اللغة مدعومة
        if lang not in self.db:
            raise ValueError(f"اللغة غير مدعومة: {lang}")
        
        # استرجاع دلالات الحرف
        return self.db[lang].get(char)
    
    def add_letter_semantics(self, letter_obj: LetterSemantics) -> None:
        """إضافة أو تحديث دلالات حرف.
        
        المعاملات:
            letter_obj: كائن LetterSemantics يحتوي على دلالات الحرف
        """
        # التأكد من أن اللغة مدعومة
        lang = letter_obj.language.value if isinstance(letter_obj.language, Language) else letter_obj.language
        if lang not in self.db:
            self.db[lang] = {}
        
        # إضافة أو تحديث دلالات الحرف
        self.db[lang][letter_obj.character] = letter_obj
    
    def get_all_letters(self, lang: Union[str, Language]) -> List[LetterSemantics]:
        """الحصول على قائمة بجميع الحروف في لغة معينة.
        
        المعاملات:
            lang: اللغة ("ar" أو "en" أو كائن Language)
            
        العائد:
            قائمة بكائنات LetterSemantics
        """
        # تحويل اللغة إلى سلسلة نصية إذا كانت كائن Language
        if isinstance(lang, Language):
            lang = lang.value
        
        # التأكد من أن اللغة مدعومة
        if lang not in self.db:
            raise ValueError(f"اللغة غير مدعومة: {lang}")
        
        # استرجاع قائمة بجميع الحروف
        return list(self.db[lang].values())
    
    def save_to_file(self, file_path: str) -> None:
        """حفظ قاعدة البيانات إلى ملف JSON.
        
        المعاملات:
            file_path: مسار الملف المراد الحفظ إليه
        """
        # تحويل قاعدة البيانات إلى قاموس
        data = {}
        for lang, letters in self.db.items():
            data[lang] = {}
            for char, letter_obj in letters.items():
                data[lang][char] = letter_obj.to_dict()
        
        # حفظ البيانات إلى ملف JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """تحميل قاعدة البيانات من ملف JSON.
        
        المعاملات:
            file_path: مسار الملف المراد التحميل منه
        """
        # تحميل البيانات من ملف JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # تحويل البيانات إلى كائنات LetterSemantics
        for lang, letters in data.items():
            if lang not in self.db:
                self.db[lang] = {}
            
            for char, letter_data in letters.items():
                self.db[lang][char] = LetterSemantics.from_dict(letter_data)

# -----------------------------------------------------------------------------
# فئة WordSemanticSynthesizer - تجميع دلالات الحروف المكونة لكلمة
# -----------------------------------------------------------------------------

@dataclass
class SemanticSynthesisResult:
    """نتيجة تجميع الدلالات لكلمة."""
    
    word: str                      # الكلمة التي تم تحليلها
    language: Language             # اللغة
    
    # نتائج التجميع بالطرق المختلفة
    sequential_narrative: Optional[Dict[str, Any]] = None  # نتيجة الآلية التسلسلية/القصصية
    overlay_blend: Optional[Dict[str, Any]] = None         # نتيجة آلية التراكب/المزيج
    semantic_axes: Optional[Dict[str, Any]] = None         # نتيجة آلية المحاور الدلالية
    dominant_symbol: Optional[Dict[str, Any]] = None       # نتيجة آلية "الرمز المهيمن"
    
    # التقييم العام
    confidence_score: float = 0.0  # درجة الثقة في التفسير (0-1)
    primary_meaning: str = ""      # المعنى الأساسي المستنبط
    semantic_tags: List[str] = field(default_factory=list)  # وسوم دلالية
    
    def get_best_synthesis(self) -> Dict[str, Any]:
        """الحصول على أفضل نتيجة تجميع بناءً على درجة الثقة.
        
        العائد:
            قاموس يمثل أفضل نتيجة تجميع
        """
        # قائمة بجميع نتائج التجميع المتاحة مع درجات الثقة
        synthesis_results = []
        
        if self.sequential_narrative:
            synthesis_results.append((self.sequential_narrative, self.sequential_narrative.get("confidence", 0.0)))
        
        if self.overlay_blend:
            synthesis_results.append((self.overlay_blend, self.overlay_blend.get("confidence", 0.0)))
        
        if self.semantic_axes:
            synthesis_results.append((self.semantic_axes, self.semantic_axes.get("confidence", 0.0)))
        
        if self.dominant_symbol:
            synthesis_results.append((self.dominant_symbol, self.dominant_symbol.get("confidence", 0.0)))
        
        # ترتيب النتائج حسب درجة الثقة (تنازلياً)
        synthesis_results.sort(key=lambda x: x[1], reverse=True)
        
        # إرجاع أفضل نتيجة، أو قاموس فارغ إذا لم تكن هناك نتائج
        return synthesis_results[0][0] if synthesis_results else {}
    
    def to_dict(self) -> Dict:
        """تحويل الكائن إلى قاموس.
        
        العائد:
            قاموس يمثل الكائن
        """
        return {
            "word": self.word,
            "language": self.language.value if isinstance(self.language, Language) else self.language,
            "sequential_narrative": self.sequential_narrative,
            "overlay_blend": self.overlay_blend,
            "semantic_axes": self.semantic_axes,
            "dominant_symbol": self.dominant_symbol,
            "confidence_score": self.confidence_score,
            "primary_meaning": self.primary_meaning,
            "semantic_tags": self.semantic_tags
        }

class WordSemanticSynthesizer:
    """تجميع دلالات الحروف المكونة لكلمة لتكوين "بصمة دلالية" أو "تفسير أولي" للكلمة."""
    
    def __init__(self, semantic_db: SemanticDatabase, knowledge_graph_manager: Optional[Any] = None):
        """تهيئة مجمع الدلالات.
        
        المعاملات:
            semantic_db: كائن SemanticDatabase يحتوي على دلالات الحروف
            knowledge_graph_manager: (اختياري) كائن KnowledgeGraphManager للمساعدة في تقييم "منطقية القصة"
        """
        self.semantic_db = semantic_db
        self.knowledge_graph_manager = knowledge_graph_manager
    
    def _apply_vowel_modification(self, letter_sem: LetterSemantics, vowel_mark: Optional[str]) -> LetterSemantics:
        """تعديل دلالات الحرف بناءً على الحركة المصاحبة.
        
        المعاملات:
            letter_sem: كائن LetterSemantics يمثل دلالات الحرف
            vowel_mark: الحركة المصاحبة للحرف (فتحة، ضمة، كسرة، سكون)
            
        العائد:
            كائن LetterSemantics معدل
        """
        # نسخة من دلالات الحرف لتجنب تعديل الأصل
        modified_letter = copy.deepcopy(letter_sem)
        
        # إذا لم تكن هناك حركة، إرجاع الدلالات كما هي
        if not vowel_mark:
            return modified_letter
        
        # تعديل الدلالات بناءً على الحركة
        if vowel_mark == 'َ':  # فتحة
            # تعزيز محور الانفتاح والارتفاع
            for axis_name in modified_letter.core_semantic_axes:
                if axis_name in ["الانفتاح", "الارتفاع"]:
                    # تعديل وصف المحور لتعكس تأثير الفتحة
                    axis = modified_letter.core_semantic_axes[axis_name]
                    axis.description = f"{axis.description} (معزز بالفتحة)"
            
            # إضافة دلالات عامة مرتبطة بالفتحة
            modified_letter.general_connotations.append("انفتاح")
            modified_letter.general_connotations.append("وضوح")
            
        elif vowel_mark == 'ُ':  # ضمة
            # تعزيز محور الارتفاع والتجمع
            for axis_name in modified_letter.core_semantic_axes:
                if axis_name in ["الارتفاع", "التجمع"]:
                    # تعديل وصف المحور لتعكس تأثير الضمة
                    axis = modified_letter.core_semantic_axes[axis_name]
                    axis.description = f"{axis.description} (معزز بالضمة)"
            
            # إضافة دلالات عامة مرتبطة بالضمة
            modified_letter.general_connotations.append("ضم")
            modified_letter.general_connotations.append("تجميع")
            
        elif vowel_mark == 'ِ':  # كسرة
            # تعزيز محور الانخفاض والانكسار
            for axis_name in modified_letter.core_semantic_axes:
                if axis_name in ["الارتفاع", "الاستقامة"]:
                    # تعديل وصف المحور لتعكس تأثير الكسرة
                    axis = modified_letter.core_semantic_axes[axis_name]
                    axis.description = f"{axis.description} (متأثر بالكسرة)"
            
            # إضافة دلالات عامة مرتبطة بالكسرة
            modified_letter.general_connotations.append("انكسار")
            modified_letter.general_connotations.append("انخفاض")
            
        elif vowel_mark == 'ْ':  # سكون
            # تعزيز محور السكون والثبات
            for axis_name in modified_letter.core_semantic_axes:
                if axis_name in ["الحركة", "التغير"]:
                    # تعديل وصف المحور لتعكس تأثير السكون
                    axis = modified_letter.core_semantic_axes[axis_name]
                    axis.description = f"{axis.description} (متأثر بالسكون)"
            
            # إضافة دلالات عامة مرتبطة بالسكون
            modified_letter.general_connotations.append("سكون")
            modified_letter.general_connotations.append("ثبات")
        
        return modified_letter
    
    def _split_word_with_diacritics(self, word: str, lang: Language) -> List[Tuple[str, Optional[str]]]:
        """تقسيم الكلمة إلى حروف وحركات.
        
        المعاملات:
            word: الكلمة المراد تقسيمها
            lang: اللغة
            
        العائد:
            قائمة من أزواج (الحرف، الحركة)
        """
        result = []
        
        # التعامل مع اللغة العربية
        if lang == Language.ARABIC:
            # تعريف الحركات العربية
            diacritics = ['َ', 'ُ', 'ِ', 'ْ', 'ّ', 'ً', 'ٌ', 'ٍ']
            
            i = 0
            while i < len(word):
                char = word[i]
                i += 1
                
                # البحث عن الحركة التالية للحرف
                vowel = None
                if i < len(word) and word[i] in diacritics:
                    vowel = word[i]
                    i += 1
                
                # إضافة الزوج (الحرف، الحركة) إلى النتيجة
                result.append((char, vowel))
        
        # التعامل مع اللغة الإنجليزية
        elif lang == Language.ENGLISH:
            # في اللغة الإنجليزية، كل حرف يعتبر وحدة منفصلة
            for char in word:
                result.append((char, None))
        
        return result
    
    def _apply_sequential_narrative_synthesis(self, letters_semantics: List[LetterSemantics], word: str, lang: Language) -> Dict[str, Any]:
        """تطبيق الآلية التسلسلية/القصصية لتجميع دلالات الحروف.
        
        المعاملات:
            letters_semantics: قائمة بكائنات LetterSemantics للحروف المكونة للكلمة
            word: الكلمة الأصلية
            lang: اللغة
            
        العائد:
            قاموس يحتوي على نتيجة التجميع
        """
        # إذا لم تكن هناك حروف، إرجاع نتيجة فارغة
        if not letters_semantics:
            return {
                "narrative": "",
                "confidence": 0.0,
                "semantic_elements": []
            }
        
        # بناء "قصة" من دلالات الحروف بالترتيب
        narrative_elements = []
        semantic_elements = []
        
        for i, letter_sem in enumerate(letters_semantics):
            # استخراج الدلالة المهيمنة للحرف
            dominant_connotation = letter_sem.get_dominant_connotation()
            
            # استخراج الدلالات العامة للحرف
            general_connotations = letter_sem.general_connotations[:2] if letter_sem.general_connotations else []
            
            # استخراج دلالات الشكل البصري للحرف
            visual_semantics = letter_sem.visual_form_semantics[:1] if letter_sem.visual_form_semantics else []
            
            # تجميع العناصر الدلالية للحرف
            letter_elements = []
            if dominant_connotation and dominant_connotation != "لا توجد دلالة مهيمنة":
                letter_elements.append(dominant_connotation)
            
            letter_elements.extend(general_connotations)
            letter_elements.extend(visual_semantics)
            
            # إضافة عنصر سردي للقصة
            if letter_elements:
                position_desc = "في البداية" if i == 0 else "ثم" if i < len(letters_semantics) - 1 else "وأخيراً"
                narrative_element = f"{position_desc} {letter_elements[0]}"
                if len(letter_elements) > 1:
                    narrative_element += f" ({', '.join(letter_elements[1:])})"
                
                narrative_elements.append(narrative_element)
                
                # إضافة العناصر الدلالية
                semantic_elements.extend(letter_elements)
        
        # بناء القصة النهائية
        narrative = " ".join(narrative_elements)
        
        # حساب درجة الثقة في القصة
        # (يمكن تحسين هذا باستخدام KnowledgeGraphManager إذا كان متاحاً)
        confidence = min(0.5 + (len(narrative_elements) / len(letters_semantics)) * 0.5, 1.0)
        
        return {
            "narrative": narrative,
            "confidence": confidence,
            "semantic_elements": semantic_elements
        }
    
    def _apply_overlay_blend_synthesis(self, letters_semantics: List[LetterSemantics], word: str, lang: Language) -> Dict[str, Any]:
        """تطبيق آلية التراكب/المزيج لتجميع دلالات الحروف.
        
        المعاملات:
            letters_semantics: قائمة بكائنات LetterSemantics للحروف المكونة للكلمة
            word: الكلمة الأصلية
            lang: اللغة
            
        العائد:
            قاموس يحتوي على نتيجة التجميع
        """
        # إذا لم تكن هناك حروف، إرجاع نتيجة فارغة
        if not letters_semantics:
            return {
                "blend_description": "",
                "confidence": 0.0,
                "semantic_elements": [],
                "weighted_connotations": {}
            }
        
        # جمع جميع الدلالات من جميع الحروف مع أوزانها
        all_connotations = defaultdict(float)
        
        # أوزان الحروف بناءً على موقعها في الكلمة
        position_weights = []
        
        # تعيين أوزان للحروف بناءً على موقعها (الأول والأخير لهما وزن أكبر)
        for i in range(len(letters_semantics)):
            if i == 0:  # الحرف الأول
                position_weights.append(1.5)
            elif i == len(letters_semantics) - 1:  # الحرف الأخير
                position_weights.append(1.2)
            else:  # الحروف الوسطى
                position_weights.append(1.0)
        
        # جمع الدلالات من جميع الحروف مع أوزانها
        for i, letter_sem in enumerate(letters_semantics):
            weight = position_weights[i]
            
            # إضافة الدلالات العامة
            for connotation in letter_sem.general_connotations:
                all_connotations[connotation] += weight
            
            # إضافة دلالات الشكل البصري
            for visual_sem in letter_sem.visual_form_semantics:
                all_connotations[visual_sem] += weight * 0.8  # وزن أقل للدلالات البصرية
            
            # إضافة المحاور الدلالية
            for axis_name, axis in letter_sem.core_semantic_axes.items():
                all_connotations[f"{axis.name}: {axis.positive_pole}"] += weight * 0.7
                all_connotations[f"{axis.name}: {axis.negative_pole}"] += weight * 0.3
        
        # ترتيب الدلالات حسب الوزن (تنازلياً)
        sorted_connotations = sorted(all_connotations.items(), key=lambda x: x[1], reverse=True)
        
        # اختيار أهم الدلالات (أعلى 5 أوزان)
        top_connotations = sorted_connotations[:5]
        
        # بناء وصف المزيج
        blend_elements = [f"{connotation} ({weight:.1f})" for connotation, weight in top_connotations]
        blend_description = "مزيج من " + " و".join(blend_elements)
        
        # حساب درجة الثقة في المزيج
        # (يمكن تحسين هذا باستخدام KnowledgeGraphManager إذا كان متاحاً)
        confidence = min(0.4 + (len(top_connotations) / 5) * 0.6, 1.0)
        
        return {
            "blend_description": blend_description,
            "confidence": confidence,
            "semantic_elements": [connotation for connotation, _ in top_connotations],
            "weighted_connotations": dict(sorted_connotations)
        }
    
    def _apply_semantic_axes_synthesis(self, letters_semantics: List[LetterSemantics], word: str, lang: Language) -> Dict[str, Any]:
        """تطبيق آلية المحاور الدلالية لتجميع دلالات الحروف.
        
        المعاملات:
            letters_semantics: قائمة بكائنات LetterSemantics للحروف المكونة للكلمة
            word: الكلمة الأصلية
            lang: اللغة
            
        العائد:
            قاموس يحتوي على نتيجة التجميع
        """
        # إذا لم تكن هناك حروف، إرجاع نتيجة فارغة
        if not letters_semantics:
            return {
                "axes_description": "",
                "confidence": 0.0,
                "semantic_axes": {}
            }
        
        # جمع جميع المحاور الدلالية من جميع الحروف
        all_axes = {}
        
        for letter_sem in letters_semantics:
            for axis_name, axis in letter_sem.core_semantic_axes.items():
                if axis_name not in all_axes:
                    all_axes[axis_name] = {
                        "name": axis.name,
                        "negative_pole": axis.negative_pole,
                        "positive_pole": axis.positive_pole,
                        "description": axis.description,
                        "occurrences": 0,
                        "intensity_sum": 0.0
                    }
                
                # زيادة عدد مرات ظهور المحور
                all_axes[axis_name]["occurrences"] += 1
                
                # إضافة شدة افتراضية (يمكن تحسين هذا لاحقاً)
                # نفترض أن الشدة هي 0.5 (محايدة) إذا لم يتم تحديدها
                all_axes[axis_name]["intensity_sum"] += 0.5
        
        # حساب متوسط الشدة لكل محور
        for axis_name in all_axes:
            all_axes[axis_name]["average_intensity"] = all_axes[axis_name]["intensity_sum"] / all_axes[axis_name]["occurrences"]
        
        # ترتيب المحاور حسب عدد مرات الظهور (تنازلياً)
        sorted_axes = sorted(all_axes.items(), key=lambda x: x[1]["occurrences"], reverse=True)
        
        # بناء وصف المحاور
        axes_descriptions = []
        for axis_name, axis_info in sorted_axes:
            intensity = axis_info["average_intensity"]
            if intensity < 0.3:
                pole_desc = f"يميل نحو {axis_info['negative_pole']}"
            elif intensity > 0.7:
                pole_desc = f"يميل نحو {axis_info['positive_pole']}"
            else:
                pole_desc = f"متوازن بين {axis_info['negative_pole']} و{axis_info['positive_pole']}"
            
            axes_descriptions.append(f"على محور {axis_info['name']}: {pole_desc}")
        
        axes_description = " | ".join(axes_descriptions)
        
        # حساب درجة الثقة في تحليل المحاور
        # (يمكن تحسين هذا باستخدام KnowledgeGraphManager إذا كان متاحاً)
        confidence = min(0.3 + (len(sorted_axes) / len(letters_semantics)) * 0.7, 1.0)
        
        return {
            "axes_description": axes_description,
            "confidence": confidence,
            "semantic_axes": {axis_name: axis_info for axis_name, axis_info in sorted_axes}
        }
    
    def _apply_dominant_symbol_synthesis(self, letters_semantics: List[LetterSemantics], word: str, lang: Language) -> Dict[str, Any]:
        """تطبيق آلية "الرمز المهيمن" لتجميع دلالات الحروف.
        
        المعاملات:
            letters_semantics: قائمة بكائنات LetterSemantics للحروف المكونة للكلمة
            word: الكلمة الأصلية
            lang: اللغة
            
        العائد:
            قاموس يحتوي على نتيجة التجميع
        """
        # إذا لم تكن هناك حروف، إرجاع نتيجة فارغة
        if not letters_semantics:
            return {
                "dominant_symbol": "",
                "confidence": 0.0,
                "explanation": ""
            }
        
        # تحديد الحرف المهيمن (الأول أو الأخير أو الأكثر بروزاً)
        dominant_letter = None
        dominant_index = -1
        
        # الحرف الأول له أولوية عالية
        dominant_letter = letters_semantics[0]
        dominant_index = 0
        
        # الحرف الأخير قد يكون مهيمناً أيضاً في بعض الحالات
        if len(letters_semantics) > 1 and len(letters_semantics[-1].general_connotations) > len(dominant_letter.general_connotations):
            dominant_letter = letters_semantics[-1]
            dominant_index = len(letters_semantics) - 1
        
        # البحث عن الحرف الأكثر بروزاً (الذي له أكبر عدد من الدلالات)
        for i, letter_sem in enumerate(letters_semantics):
            if len(letter_sem.general_connotations) > len(dominant_letter.general_connotations):
                dominant_letter = letter_sem
                dominant_index = i
        
        # استخراج الدلالة المهيمنة للحرف المهيمن
        dominant_connotation = dominant_letter.get_dominant_connotation()
        
        # بناء وصف الرمز المهيمن
        position_desc = "الأول" if dominant_index == 0 else "الأخير" if dominant_index == len(letters_semantics) - 1 else f"في الموقع {dominant_index + 1}"
        explanation = f"الحرف المهيمن هو '{dominant_letter.character}' ({position_desc}) ويرمز إلى {dominant_connotation}"
        
        if dominant_letter.general_connotations:
            explanation += f". دلالاته العامة: {', '.join(dominant_letter.general_connotations[:3])}"
        
        # حساب درجة الثقة في تحليل الرمز المهيمن
        # (يمكن تحسين هذا باستخدام KnowledgeGraphManager إذا كان متاحاً)
        confidence = 0.6  # درجة ثقة متوسطة لهذه الطريقة
        
        return {
            "dominant_symbol": dominant_letter.character,
            "dominant_connotation": dominant_connotation,
            "confidence": confidence,
            "explanation": explanation
        }
    
    def synthesize_word_meaning(self, word: str, lang: Union[str, Language], context: Optional[Dict] = None, methods: Union[List[SemanticSynthesisMethod], SemanticSynthesisMethod] = SemanticSynthesisMethod.ALL) -> SemanticSynthesisResult:
        """تجميع دلالات الحروف المكونة لكلمة لتكوين "بصمة دلالية" أو "تفسير أولي" للكلمة.
        
        المعاملات:
            word: الكلمة المراد تحليلها
            lang: اللغة ("ar" أو "en" أو كائن Language)
            context: (اختياري) سياق إضافي للتحليل
            methods: طرق التجميع المراد استخدامها
            
        العائد:
            كائن SemanticSynthesisResult يحتوي على نتيجة التجميع
        """
        # تحويل اللغة إلى كائن Language
        if isinstance(lang, str):
            lang = Language.from_string(lang)
        
        # تحويل methods إلى قائمة إذا كان قيمة واحدة
        if isinstance(methods, SemanticSynthesisMethod):
            if methods == SemanticSynthesisMethod.ALL:
                methods = [
                    SemanticSynthesisMethod.SEQUENTIAL_NARRATIVE,
                    SemanticSynthesisMethod.OVERLAY_BLEND,
                    SemanticSynthesisMethod.SEMANTIC_AXES,
                    SemanticSynthesisMethod.DOMINANT_SYMBOL
                ]
            else:
                methods = [methods]
        
        # تقسيم الكلمة إلى حروف وحركات
        letters_with_diacritics = self._split_word_with_diacritics(word, lang)
        
        # استرجاع دلالات كل حرف مع تطبيق تعديل الحركات
        letters_semantics = []
        for char, vowel in letters_with_diacritics:
            letter_sem = self.semantic_db.get_letter_semantics(char, lang)
            if letter_sem:
                # تطبيق تعديل الحركات
                modified_letter_sem = self._apply_vowel_modification(letter_sem, vowel)
                letters_semantics.append(modified_letter_sem)
        
        # تهيئة نتيجة التجميع
        result = SemanticSynthesisResult(
            word=word,
            language=lang
        )
        
        # تطبيق طرق التجميع المطلوبة
        if SemanticSynthesisMethod.SEQUENTIAL_NARRATIVE in methods:
            result.sequential_narrative = self._apply_sequential_narrative_synthesis(letters_semantics, word, lang)
        
        if SemanticSynthesisMethod.OVERLAY_BLEND in methods:
            result.overlay_blend = self._apply_overlay_blend_synthesis(letters_semantics, word, lang)
        
        if SemanticSynthesisMethod.SEMANTIC_AXES in methods:
            result.semantic_axes = self._apply_semantic_axes_synthesis(letters_semantics, word, lang)
        
        if SemanticSynthesisMethod.DOMINANT_SYMBOL in methods:
            result.dominant_symbol = self._apply_dominant_symbol_synthesis(letters_semantics, word, lang)
        
        # تحديد المعنى الأساسي والوسوم الدلالية
        self._determine_primary_meaning_and_tags(result)
        
        # حساب درجة الثقة الإجمالية
        self._calculate_overall_confidence(result)
        
        return result
    
    def _determine_primary_meaning_and_tags(self, result: SemanticSynthesisResult) -> None:
        """تحديد المعنى الأساسي والوسوم الدلالية لنتيجة التجميع.
        
        المعاملات:
            result: كائن SemanticSynthesisResult المراد تحديد المعنى الأساسي والوسوم الدلالية له
        """
        # جمع جميع العناصر الدلالية من جميع طرق التجميع
        all_semantic_elements = set()
        
        if result.sequential_narrative:
            all_semantic_elements.update(result.sequential_narrative.get("semantic_elements", []))
        
        if result.overlay_blend:
            all_semantic_elements.update(result.overlay_blend.get("semantic_elements", []))
        
        if result.semantic_axes:
            for axis_info in result.semantic_axes.get("semantic_axes", {}).values():
                intensity = axis_info.get("average_intensity", 0.5)
                if intensity < 0.3:
                    all_semantic_elements.add(axis_info["negative_pole"])
                elif intensity > 0.7:
                    all_semantic_elements.add(axis_info["positive_pole"])
                else:
                    all_semantic_elements.add(f"{axis_info['negative_pole']}-{axis_info['positive_pole']}")
        
        if result.dominant_symbol:
            all_semantic_elements.add(result.dominant_symbol.get("dominant_connotation", ""))
        
        # تحديد الوسوم الدلالية
        result.semantic_tags = list(all_semantic_elements)
        
        # تحديد المعنى الأساسي
        if result.sequential_narrative and result.sequential_narrative.get("confidence", 0.0) >= 0.6:
            # استخدام القصة التسلسلية إذا كانت درجة الثقة عالية
            result.primary_meaning = result.sequential_narrative.get("narrative", "")
        elif result.overlay_blend and result.overlay_blend.get("confidence", 0.0) >= 0.6:
            # استخدام وصف المزيج إذا كانت درجة الثقة عالية
            result.primary_meaning = result.overlay_blend.get("blend_description", "")
        elif result.dominant_symbol:
            # استخدام شرح الرمز المهيمن كملاذ أخير
            result.primary_meaning = result.dominant_symbol.get("explanation", "")
        else:
            # استخدام الوسوم الدلالية إذا لم تكن هناك طريقة أخرى
            result.primary_meaning = "مزيج من " + " و".join(result.semantic_tags[:3]) if result.semantic_tags else ""
    
    def _calculate_overall_confidence(self, result: SemanticSynthesisResult) -> None:
        """حساب درجة الثقة الإجمالية لنتيجة التجميع.
        
        المعاملات:
            result: كائن SemanticSynthesisResult المراد حساب درجة الثقة الإجمالية له
        """
        # جمع درجات الثقة من جميع طرق التجميع
        confidence_scores = []
        
        if result.sequential_narrative:
            confidence_scores.append(result.sequential_narrative.get("confidence", 0.0))
        
        if result.overlay_blend:
            confidence_scores.append(result.overlay_blend.get("confidence", 0.0))
        
        if result.semantic_axes:
            confidence_scores.append(result.semantic_axes.get("confidence", 0.0))
        
        if result.dominant_symbol:
            confidence_scores.append(result.dominant_symbol.get("confidence", 0.0))
        
        # حساب المتوسط المرجح لدرجات الثقة
        if confidence_scores:
            # إعطاء وزن أكبر لأعلى درجة ثقة
            max_confidence = max(confidence_scores)
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            result.confidence_score = 0.7 * max_confidence + 0.3 * avg_confidence
        else:
            result.confidence_score = 0.0

# -----------------------------------------------------------------------------
# فئة SymbolicInterpretationEngine - محرك تفسير الرموز ودلالات الحروف
# -----------------------------------------------------------------------------

class SymbolicInterpretationEngine:
    """محرك تفسير الرموز ودلالات الحروف - الفئة الرئيسية للوحدة."""
    
    def __init__(self, semantic_db: Optional[SemanticDatabase] = None, word_synthesizer: Optional[WordSemanticSynthesizer] = None, data_file: Optional[str] = None):
        """تهيئة محرك تفسير الرموز.
        
        المعاملات:
            semantic_db: (اختياري) كائن SemanticDatabase يحتوي على دلالات الحروف
            word_synthesizer: (اختياري) كائن WordSemanticSynthesizer لتجميع دلالات الحروف
            data_file: (اختياري) مسار ملف JSON يحتوي على بيانات الحروف
        """
        # تهيئة قاعدة البيانات الدلالية
        if semantic_db:
            self.semantic_db = semantic_db
        else:
            self.semantic_db = SemanticDatabase(data_file)
        
        # تهيئة مجمع الدلالات
        if word_synthesizer:
            self.word_synthesizer = word_synthesizer
        else:
            self.word_synthesizer = WordSemanticSynthesizer(self.semantic_db)
    
    def get_letter_symbol_details(self, char: str, lang: Union[str, Language]) -> Optional[LetterSemantics]:
        """الحصول على تفاصيل دلالات حرف معين.
        
        المعاملات:
            char: الحرف المراد الحصول على دلالاته
            lang: اللغة ("ar" أو "en" أو كائن Language)
            
        العائد:
            كائن LetterSemantics أو None إذا لم يتم العثور على الحرف
        """
        return self.semantic_db.get_letter_semantics(char, lang)
    
    def interpret_word(self, word: str, lang: Union[str, Language], context: Optional[Dict] = None, methods: Union[List[SemanticSynthesisMethod], SemanticSynthesisMethod] = SemanticSynthesisMethod.ALL) -> SemanticSynthesisResult:
        """تفسير كلمة بناءً على دلالات حروفها.
        
        المعاملات:
            word: الكلمة المراد تفسيرها
            lang: اللغة ("ar" أو "en" أو كائن Language)
            context: (اختياري) سياق إضافي للتفسير
            methods: طرق التجميع المراد استخدامها
            
        العائد:
            كائن SemanticSynthesisResult يحتوي على نتيجة التفسير
        """
        return self.word_synthesizer.synthesize_word_meaning(word, lang, context, methods)
    
    def interpret_shape_equation_semantics(self, parsed_shape_representation: List[Dict[str, Any]], lang: Union[str, Language]) -> List[Dict[str, Any]]:
        """تفسير دلالات معادلات الشكل.
        
        المعاملات:
            parsed_shape_representation: تمثيل الشكل المُحلل من AdvancedShapeEquationParser
            lang: اللغة ("ar" أو "en" أو كائن Language)
            
        العائد:
            قائمة من القواميس تحتوي على تفسيرات دلالية لمكونات الشكل
        """
        # تحويل اللغة إلى كائن Language
        if isinstance(lang, str):
            lang = Language.from_string(lang)
        
        # قائمة لتخزين التفسيرات الدلالية
        semantic_interpretations = []
        
        # تفسير كل مكون شكل
        for shape_component in parsed_shape_representation:
            # استخراج نوع الشكل
            shape_type = shape_component.get("type", "")
            
            # تفسير نوع الشكل
            shape_type_interpretation = None
            if shape_type:
                # تحويل نوع الشكل إلى كلمة قابلة للتفسير
                interpretable_word = shape_type.lower().replace("_", " ")
                shape_type_interpretation = self.interpret_word(interpretable_word, "en")
            
            # استخراج المعاملات
            params = shape_component.get("params", {})
            
            # تفسير المعاملات
            params_interpretations = {}
            for param_name, param_value in params.items():
                # تحويل اسم المعامل إلى كلمة قابلة للتفسير
                interpretable_param_name = param_name.lower().replace("_", " ")
                param_interpretation = self.interpret_word(interpretable_param_name, "en")
                params_interpretations[param_name] = param_interpretation.to_dict()
            
            # تجميع التفسير الدلالي للمكون
            component_interpretation = {
                "component_type": shape_type,
                "component_type_interpretation": shape_type_interpretation.to_dict() if shape_type_interpretation else None,
                "params_interpretations": params_interpretations,
                "semantic_tags": []
            }
            
            # استخراج الوسوم الدلالية من تفسير نوع الشكل
            if shape_type_interpretation:
                component_interpretation["semantic_tags"].extend(shape_type_interpretation.semantic_tags)
            
            # إضافة التفسير إلى القائمة
            semantic_interpretations.append(component_interpretation)
        
        return semantic_interpretations
    
    def interpret_dream_symbols(self, dream_elements: List[str], context: Optional[Dict] = None, lang: Union[str, Language] = Language.ARABIC) -> Dict[str, Any]:
        """تفسير رموز الأحلام.
        
        المعاملات:
            dream_elements: قائمة بعناصر الحلم المراد تفسيرها
            context: (اختياري) سياق إضافي للتفسير
            lang: اللغة ("ar" أو "en" أو كائن Language)
            
        العائد:
            قاموس يحتوي على تفسيرات رموز الحلم
        """
        # تحويل اللغة إلى كائن Language
        if isinstance(lang, str):
            lang = Language.from_string(lang)
        
        # قاموس لتخزين تفسيرات رموز الحلم
        dream_interpretations = {}
        
        # تفسير كل عنصر من عناصر الحلم
        for element in dream_elements:
            # تفسير العنصر بناءً على دلالات حروفه
            element_interpretation = self.interpret_word(element, lang, context)
            
            # إضافة التفسير إلى القاموس
            dream_interpretations[element] = element_interpretation.to_dict()
        
        return {
            "dream_elements": dream_elements,
            "interpretations": dream_interpretations,
            "overall_interpretation": "تفسير الحلم بشكل عام يعتمد على تفاعل جميع العناصر معاً"
        }
    
    def save_database(self, file_path: str) -> None:
        """حفظ قاعدة البيانات الدلالية إلى ملف.
        
        المعاملات:
            file_path: مسار الملف المراد الحفظ إليه
        """
        self.semantic_db.save_to_file(file_path)
    
    def load_database(self, file_path: str) -> None:
        """تحميل قاعدة البيانات الدلالية من ملف.
        
        المعاملات:
            file_path: مسار الملف المراد التحميل منه
        """
        self.semantic_db.load_from_file(file_path)

# -----------------------------------------------------------------------------
# دوال مساعدة على مستوى الوحدة
# -----------------------------------------------------------------------------

def get_letter_semantics_for_language(lang: Union[str, Language]) -> Dict[str, Dict]:
    """الحصول على دلالات جميع الحروف في لغة معينة.
    
    المعاملات:
        lang: اللغة ("ar" أو "en" أو كائن Language)
        
    العائد:
        قاموس يحتوي على دلالات جميع الحروف في اللغة المحددة
    """
    # تهيئة قاعدة البيانات الدلالية
    semantic_db = SemanticDatabase()
    
    # تحويل اللغة إلى كائن Language
    if isinstance(lang, str):
        lang = Language.from_string(lang)
    
    # الحصول على قائمة بجميع الحروف في اللغة المحددة
    letters = semantic_db.get_all_letters(lang)
    
    # تحويل القائمة إلى قاموس
    result = {}
    for letter in letters:
        result[letter.character] = letter.to_dict()
    
    return result

def analyze_word_semantics(word: str, lang: Union[str, Language]) -> Dict[str, Any]:
    """تحليل دلالات كلمة بناءً على حروفها.
    
    المعاملات:
        word: الكلمة المراد تحليلها
        lang: اللغة ("ar" أو "en" أو كائن Language)
        
    العائد:
        قاموس يحتوي على نتيجة التحليل
    """
    # تهيئة محرك تفسير الرموز
    engine = SymbolicInterpretationEngine()
    
    # تفسير الكلمة
    result = engine.interpret_word(word, lang)
    
    # تحويل النتيجة إلى قاموس
    return result.to_dict()

def compare_words_semantically(word1: str, word2: str, lang: Union[str, Language]) -> Dict[str, Any]:
    """مقارنة كلمتين دلالياً.
    
    المعاملات:
        word1: الكلمة الأولى
        word2: الكلمة الثانية
        lang: اللغة ("ar" أو "en" أو كائن Language)
        
    العائد:
        قاموس يحتوي على نتيجة المقارنة
    """
    # تهيئة محرك تفسير الرموز
    engine = SymbolicInterpretationEngine()
    
    # تفسير الكلمتين
    result1 = engine.interpret_word(word1, lang)
    result2 = engine.interpret_word(word2, lang)
    
    # حساب التشابه الدلالي
    common_tags = set(result1.semantic_tags).intersection(set(result2.semantic_tags))
    similarity_score = len(common_tags) / max(len(result1.semantic_tags), len(result2.semantic_tags)) if max(len(result1.semantic_tags), len(result2.semantic_tags)) > 0 else 0.0
    
    return {
        "word1": word1,
        "word2": word2,
        "word1_semantics": result1.to_dict(),
        "word2_semantics": result2.to_dict(),
        "common_semantic_tags": list(common_tags),
        "similarity_score": similarity_score,
        "comparison_summary": f"الكلمتان متشابهتان دلالياً بنسبة {similarity_score:.2f}"
    }

# -----------------------------------------------------------------------------
# نقطة الدخول للوحدة
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # مثال على استخدام الوحدة
    engine = SymbolicInterpretationEngine()
    
    # تفسير كلمة عربية
    arabic_word = "جبل"
    arabic_result = engine.interpret_word(arabic_word, Language.ARABIC)
    print(f"تفسير كلمة '{arabic_word}':")
    print(f"المعنى الأساسي: {arabic_result.primary_meaning}")
    print(f"الوسوم الدلالية: {', '.join(arabic_result.semantic_tags)}")
    print(f"درجة الثقة: {arabic_result.confidence_score:.2f}")
    print()
    
    # تفسير كلمة إنجليزية
    english_word = "mountain"
    english_result = engine.interpret_word(english_word, Language.ENGLISH)
    print(f"تفسير كلمة '{english_word}':")
    print(f"المعنى الأساسي: {english_result.primary_meaning}")
    print(f"الوسوم الدلالية: {', '.join(english_result.semantic_tags)}")
    print(f"درجة الثقة: {english_result.confidence_score:.2f}")
