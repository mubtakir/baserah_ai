#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محلل النحو العربي

هذا الملف يحتوي على فئات تحليل النحو العربي، بما في ذلك تحليل الجمل
وتحديد أجزاء الكلام والإعراب.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import re
import os
import json
import logging
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from enum import Enum, auto
import sys

# إضافة المسار إلى حزمة utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import normalize_arabic_text, tokenize_arabic_text
from utils.data_loader import load_arabic_pos_tags, load_arabic_grammar_rules
from morphology.root_extractor import ArabicRootExtractor, RootExtractionStrategy

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arabic_nlp.syntax.syntax_analyzer')


class ArabicPOSTag(Enum):
    """أنواع أجزاء الكلام في اللغة العربية."""
    NOUN = auto()  # اسم
    VERB = auto()  # فعل
    PARTICLE = auto()  # حرف
    ADJECTIVE = auto()  # صفة
    ADVERB = auto()  # ظرف
    PRONOUN = auto()  # ضمير
    PROPER_NOUN = auto()  # اسم علم
    NUMBER = auto()  # عدد
    PUNCTUATION = auto()  # علامة ترقيم
    OTHER = auto()  # أخرى


class ArabicVerbType(Enum):
    """أنواع الأفعال في اللغة العربية."""
    PAST = auto()  # ماضي
    PRESENT = auto()  # مضارع
    IMPERATIVE = auto()  # أمر


class ArabicNounType(Enum):
    """أنواع الأسماء في اللغة العربية."""
    CONCRETE = auto()  # اسم ذات
    ABSTRACT = auto()  # اسم معنى
    PROPER = auto()  # اسم علم
    GERUND = auto()  # مصدر
    DERIVED = auto()  # مشتق


class ArabicGender(Enum):
    """الجنس في اللغة العربية."""
    MASCULINE = auto()  # مذكر
    FEMININE = auto()  # مؤنث


class ArabicNumber(Enum):
    """العدد في اللغة العربية."""
    SINGULAR = auto()  # مفرد
    DUAL = auto()  # مثنى
    PLURAL = auto()  # جمع


class ArabicCase(Enum):
    """حالات الإعراب في اللغة العربية."""
    NOMINATIVE = auto()  # مرفوع
    ACCUSATIVE = auto()  # منصوب
    GENITIVE = auto()  # مجرور


class ArabicDefiniteness(Enum):
    """التعريف والتنكير في اللغة العربية."""
    DEFINITE = auto()  # معرفة
    INDEFINITE = auto()  # نكرة


class ArabicToken:
    """رمز (كلمة) في النص العربي."""
    
    def __init__(self, text: str, position: int):
        """
        تهيئة الرمز.
        
        Args:
            text: نص الرمز
            position: موقع الرمز في النص
        """
        self.text = text
        self.position = position
        self.normalized_text = normalize_arabic_text(text)
        self.pos_tag = None  # نوع جزء الكلام
        self.lemma = None  # الجذع
        self.root = None  # الجذر
        self.morphological_features = {}  # الخصائص الصرفية
        self.syntactic_features = {}  # الخصائص النحوية
        self.dependencies = []  # العلاقات النحوية
    
    def __str__(self) -> str:
        """تمثيل الرمز كنص."""
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل الرمز إلى قاموس."""
        return {
            "text": self.text,
            "position": self.position,
            "normalized_text": self.normalized_text,
            "pos_tag": self.pos_tag.name if self.pos_tag else None,
            "lemma": self.lemma,
            "root": self.root,
            "morphological_features": self.morphological_features,
            "syntactic_features": self.syntactic_features,
            "dependencies": self.dependencies
        }


class ArabicSentence:
    """جملة في النص العربي."""
    
    def __init__(self, text: str, position: int):
        """
        تهيئة الجملة.
        
        Args:
            text: نص الجملة
            position: موقع الجملة في النص
        """
        self.text = text
        self.position = position
        self.tokens = []  # الرموز (الكلمات) في الجملة
        self.sentence_type = None  # نوع الجملة (اسمية، فعلية، إلخ)
        self.subject = None  # المسند إليه (المبتدأ أو الفاعل)
        self.predicate = None  # المسند (الخبر أو الفعل والمفعول به)
        self.parsed = False  # هل تم تحليل الجملة
    
    def __str__(self) -> str:
        """تمثيل الجملة كنص."""
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل الجملة إلى قاموس."""
        return {
            "text": self.text,
            "position": self.position,
            "tokens": [token.to_dict() for token in self.tokens],
            "sentence_type": self.sentence_type,
            "subject": self.subject.to_dict() if self.subject else None,
            "predicate": self.predicate.to_dict() if self.predicate else None,
            "parsed": self.parsed
        }


class ArabicPOSTagger:
    """محدد أجزاء الكلام في اللغة العربية."""
    
    def __init__(self, pos_tags_file: str = None):
        """
        تهيئة محدد أجزاء الكلام.
        
        Args:
            pos_tags_file: مسار ملف أجزاء الكلام
        """
        self.logger = logging.getLogger('arabic_nlp.syntax.pos_tagger')
        
        # تحميل بيانات أجزاء الكلام
        self.pos_tags_data = load_arabic_pos_tags(pos_tags_file)
        
        # إنشاء قواميس للبحث السريع
        self.verbs = self.pos_tags_data.get('verbs', {})
        self.nouns = self.pos_tags_data.get('nouns', {})
        self.particles = self.pos_tags_data.get('particles', {})
        self.adjectives = self.pos_tags_data.get('adjectives', {})
        self.adverbs = self.pos_tags_data.get('adverbs', {})
        self.pronouns = self.pos_tags_data.get('pronouns', {})
        self.proper_nouns = self.pos_tags_data.get('proper_nouns', {})
        self.numbers = self.pos_tags_data.get('numbers', {})
        self.punctuations = self.pos_tags_data.get('punctuations', {})
        
        # إنشاء أنماط للتعرف على أجزاء الكلام
        self.verb_patterns = self.pos_tags_data.get('verb_patterns', [])
        self.noun_patterns = self.pos_tags_data.get('noun_patterns', [])
        self.adjective_patterns = self.pos_tags_data.get('adjective_patterns', [])
        
        # إنشاء مستخرج الجذور
        self.root_extractor = ArabicRootExtractor(
            strategy=RootExtractionStrategy.HYBRID
        )
    
    def tag(self, token: ArabicToken) -> ArabicPOSTag:
        """
        تحديد نوع جزء الكلام للرمز.
        
        Args:
            token: الرمز المراد تحديد نوع جزء الكلام له
            
        Returns:
            نوع جزء الكلام
        """
        text = token.normalized_text
        
        # التحقق من علامات الترقيم
        if text in self.punctuations:
            token.pos_tag = ArabicPOSTag.PUNCTUATION
            return token.pos_tag
        
        # التحقق من الأعداد
        if text in self.numbers or text.isdigit():
            token.pos_tag = ArabicPOSTag.NUMBER
            return token.pos_tag
        
        # التحقق من الضمائر
        if text in self.pronouns:
            token.pos_tag = ArabicPOSTag.PRONOUN
            token.morphological_features = self.pronouns[text]
            return token.pos_tag
        
        # التحقق من الحروف
        if text in self.particles:
            token.pos_tag = ArabicPOSTag.PARTICLE
            token.morphological_features = self.particles[text]
            return token.pos_tag
        
        # التحقق من الظروف
        if text in self.adverbs:
            token.pos_tag = ArabicPOSTag.ADVERB
            token.morphological_features = self.adverbs[text]
            return token.pos_tag
        
        # التحقق من أسماء الأعلام
        if text in self.proper_nouns:
            token.pos_tag = ArabicPOSTag.PROPER_NOUN
            token.morphological_features = self.proper_nouns[text]
            return token.pos_tag
        
        # التحقق من الأفعال
        if text in self.verbs:
            token.pos_tag = ArabicPOSTag.VERB
            token.morphological_features = self.verbs[text]
            # استخراج الجذر
            token.root = self.root_extractor.get_root(text)
            return token.pos_tag
        
        # التحقق من الصفات
        if text in self.adjectives:
            token.pos_tag = ArabicPOSTag.ADJECTIVE
            token.morphological_features = self.adjectives[text]
            # استخراج الجذر
            token.root = self.root_extractor.get_root(text)
            return token.pos_tag
        
        # التحقق من الأسماء
        if text in self.nouns:
            token.pos_tag = ArabicPOSTag.NOUN
            token.morphological_features = self.nouns[text]
            # استخراج الجذر
            token.root = self.root_extractor.get_root(text)
            return token.pos_tag
        
        # إذا لم يتم التعرف على الكلمة، نستخدم الأنماط
        # التحقق من أنماط الأفعال
        for pattern in self.verb_patterns:
            if re.match(pattern, text):
                token.pos_tag = ArabicPOSTag.VERB
                # استخراج الجذر
                token.root = self.root_extractor.get_root(text)
                return token.pos_tag
        
        # التحقق من أنماط الصفات
        for pattern in self.adjective_patterns:
            if re.match(pattern, text):
                token.pos_tag = ArabicPOSTag.ADJECTIVE
                # استخراج الجذر
                token.root = self.root_extractor.get_root(text)
                return token.pos_tag
        
        # التحقق من أنماط الأسماء
        for pattern in self.noun_patterns:
            if re.match(pattern, text):
                token.pos_tag = ArabicPOSTag.NOUN
                # استخراج الجذر
                token.root = self.root_extractor.get_root(text)
                return token.pos_tag
        
        # إذا لم يتم التعرف على الكلمة، نفترض أنها اسم
        token.pos_tag = ArabicPOSTag.NOUN
        # استخراج الجذر
        token.root = self.root_extractor.get_root(text)
        
        return token.pos_tag


class ArabicSyntaxAnalyzer:
    """محلل النحو العربي."""
    
    def __init__(self, grammar_rules_file: str = None, pos_tags_file: str = None):
        """
        تهيئة محلل النحو.
        
        Args:
            grammar_rules_file: مسار ملف قواعد النحو
            pos_tags_file: مسار ملف أجزاء الكلام
        """
        self.logger = logging.getLogger('arabic_nlp.syntax.syntax_analyzer')
        
        # تحميل قواعد النحو
        self.grammar_rules = load_arabic_grammar_rules(grammar_rules_file)
        
        # إنشاء محدد أجزاء الكلام
        self.pos_tagger = ArabicPOSTagger(pos_tags_file)
    
    def tokenize(self, text: str) -> List[ArabicToken]:
        """
        تقسيم النص إلى رموز (كلمات).
        
        Args:
            text: النص المراد تقسيمه
            
        Returns:
            قائمة الرموز
        """
        tokens_text = tokenize_arabic_text(text)
        tokens = []
        
        for i, token_text in enumerate(tokens_text):
            token = ArabicToken(token_text, i)
            tokens.append(token)
        
        return tokens
    
    def tag_tokens(self, tokens: List[ArabicToken]) -> List[ArabicToken]:
        """
        تحديد أجزاء الكلام للرموز.
        
        Args:
            tokens: قائمة الرموز
            
        Returns:
            قائمة الرموز مع تحديد أجزاء الكلام
        """
        for token in tokens:
            self.pos_tagger.tag(token)
        
        return tokens
    
    def parse_sentence(self, sentence: ArabicSentence) -> ArabicSentence:
        """
        تحليل الجملة نحوياً.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            الجملة بعد التحليل
        """
        # تقسيم الجملة إلى رموز
        tokens = self.tokenize(sentence.text)
        sentence.tokens = tokens
        
        # تحديد أجزاء الكلام للرموز
        self.tag_tokens(sentence.tokens)
        
        # تحديد نوع الجملة (اسمية أو فعلية)
        sentence_type = self._determine_sentence_type(sentence.tokens)
        sentence.sentence_type = sentence_type
        
        # تحديد المسند إليه والمسند
        if sentence_type == "nominal":  # جملة اسمية
            self._parse_nominal_sentence(sentence)
        elif sentence_type == "verbal":  # جملة فعلية
            self._parse_verbal_sentence(sentence)
        
        sentence.parsed = True
        
        return sentence
    
    def _determine_sentence_type(self, tokens: List[ArabicToken]) -> str:
        """
        تحديد نوع الجملة (اسمية أو فعلية).
        
        Args:
            tokens: قائمة الرموز
            
        Returns:
            نوع الجملة
        """
        # إذا كانت الجملة تبدأ بفعل، فهي جملة فعلية
        if tokens and tokens[0].pos_tag == ArabicPOSTag.VERB:
            return "verbal"
        
        # وإلا فهي جملة اسمية
        return "nominal"
    
    def _parse_nominal_sentence(self, sentence: ArabicSentence) -> None:
        """
        تحليل الجملة الاسمية.
        
        Args:
            sentence: الجملة المراد تحليلها
        """
        tokens = sentence.tokens
        
        # البحث عن المبتدأ (المسند إليه)
        subject_index = -1
        for i, token in enumerate(tokens):
            if token.pos_tag in [ArabicPOSTag.NOUN, ArabicPOSTag.PROPER_NOUN, ArabicPOSTag.PRONOUN]:
                subject_index = i
                break
        
        if subject_index >= 0:
            sentence.subject = tokens[subject_index]
            tokens[subject_index].syntactic_features["role"] = "subject"
        
        # البحث عن الخبر (المسند)
        predicate_index = -1
        for i in range(subject_index + 1, len(tokens)):
            token = tokens[i]
            if token.pos_tag in [ArabicPOSTag.NOUN, ArabicPOSTag.ADJECTIVE, ArabicPOSTag.VERB]:
                predicate_index = i
                break
        
        if predicate_index >= 0:
            sentence.predicate = tokens[predicate_index]
            tokens[predicate_index].syntactic_features["role"] = "predicate"
    
    def _parse_verbal_sentence(self, sentence: ArabicSentence) -> None:
        """
        تحليل الجملة الفعلية.
        
        Args:
            sentence: الجملة المراد تحليلها
        """
        tokens = sentence.tokens
        
        # البحث عن الفعل (المسند)
        verb_index = -1
        for i, token in enumerate(tokens):
            if token.pos_tag == ArabicPOSTag.VERB:
                verb_index = i
                break
        
        if verb_index >= 0:
            sentence.predicate = tokens[verb_index]
            tokens[verb_index].syntactic_features["role"] = "verb"
        
        # البحث عن الفاعل (المسند إليه)
        subject_index = -1
        for i in range(verb_index + 1, len(tokens)):
            token = tokens[i]
            if token.pos_tag in [ArabicPOSTag.NOUN, ArabicPOSTag.PROPER_NOUN, ArabicPOSTag.PRONOUN]:
                subject_index = i
                break
        
        if subject_index >= 0:
            sentence.subject = tokens[subject_index]
            tokens[subject_index].syntactic_features["role"] = "subject"
        
        # البحث عن المفعول به
        object_index = -1
        for i in range(subject_index + 1, len(tokens)):
            token = tokens[i]
            if token.pos_tag in [ArabicPOSTag.NOUN, ArabicPOSTag.PROPER_NOUN, ArabicPOSTag.PRONOUN]:
                object_index = i
                break
        
        if object_index >= 0:
            tokens[object_index].syntactic_features["role"] = "object"
    
    def analyze(self, text: str) -> List[ArabicSentence]:
        """
        تحليل النص نحوياً.
        
        Args:
            text: النص المراد تحليله
            
        Returns:
            قائمة الجمل بعد التحليل
        """
        # تقسيم النص إلى جمل
        sentences_text = self._split_into_sentences(text)
        sentences = []
        
        for i, sentence_text in enumerate(sentences_text):
            sentence = ArabicSentence(sentence_text, i)
            self.parse_sentence(sentence)
            sentences.append(sentence)
        
        return sentences
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        تقسيم النص إلى جمل.
        
        Args:
            text: النص المراد تقسيمه
            
        Returns:
            قائمة الجمل
        """
        # تقسيم النص باستخدام علامات الترقيم
        sentences = re.split(r'[.!?؟]', text)
        # إزالة الجمل الفارغة
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


# --- اختبارات ---
if __name__ == "__main__":
    # إنشاء محلل النحو
    analyzer = ArabicSyntaxAnalyzer()
    
    # اختبار تحليل بعض الجمل
    test_sentences = [
        "الولد يلعب في الحديقة.",
        "ذهب محمد إلى المدرسة.",
        "الكتاب على الطاولة.",
        "يدرس الطلاب في الجامعة.",
        "القطة الصغيرة تنام على السرير."
    ]
    
    for test_sentence in test_sentences:
        print(f"\nتحليل الجملة: {test_sentence}")
        sentences = analyzer.analyze(test_sentence)
        
        for sentence in sentences:
            print(f"نوع الجملة: {sentence.sentence_type}")
            
            if sentence.subject:
                print(f"المسند إليه: {sentence.subject.text} ({sentence.subject.pos_tag.name})")
            
            if sentence.predicate:
                print(f"المسند: {sentence.predicate.text} ({sentence.predicate.pos_tag.name})")
            
            print("تحليل الكلمات:")
            for token in sentence.tokens:
                pos_tag = token.pos_tag.name if token.pos_tag else "غير معروف"
                role = token.syntactic_features.get("role", "غير معروف")
                root = token.root if token.root else "غير معروف"
                print(f"  - {token.text}: {pos_tag}, الدور: {role}, الجذر: {root}")
