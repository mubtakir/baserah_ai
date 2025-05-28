#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محلل البلاغة العربية

هذا الملف يحتوي على فئات تحليل البلاغة العربية، بما في ذلك تحديد
الأساليب البلاغية والصور البيانية والمحسنات البديعية.

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
from utils.data_loader import load_arabic_rhetoric_patterns
from syntax.syntax_analyzer import ArabicSyntaxAnalyzer, ArabicSentence, ArabicToken, ArabicPOSTag

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arabic_nlp.rhetoric.rhetoric_analyzer')


class RhetoricType(Enum):
    """أنواع الأساليب البلاغية."""
    SIMILE = auto()  # تشبيه
    METAPHOR = auto()  # استعارة
    METONYMY = auto()  # كناية
    ALLITERATION = auto()  # جناس
    ANTITHESIS = auto()  # طباق
    PARONOMASIA = auto()  # جناس
    ASSONANCE = auto()  # سجع
    HYPERBOLE = auto()  # مبالغة
    PERSONIFICATION = auto()  # تشخيص
    INTERROGATION = auto()  # استفهام
    EXCLAMATION = auto()  # تعجب
    IMPERATIVE = auto()  # أمر
    NEGATION = auto()  # نفي
    EMPHASIS = auto()  # توكيد
    OTHER = auto()  # أخرى


class RhetoricCategory(Enum):
    """فئات البلاغة العربية."""
    BAYAN = auto()  # علم البيان
    MAANI = auto()  # علم المعاني
    BADI = auto()  # علم البديع


class RhetoricElement:
    """عنصر بلاغي في النص."""
    
    def __init__(self, 
                 text: str, 
                 rhetoric_type: RhetoricType, 
                 category: RhetoricCategory,
                 start_pos: int, 
                 end_pos: int):
        """
        تهيئة العنصر البلاغي.
        
        Args:
            text: نص العنصر البلاغي
            rhetoric_type: نوع الأسلوب البلاغي
            category: فئة البلاغة
            start_pos: موقع بداية العنصر في النص
            end_pos: موقع نهاية العنصر في النص
        """
        self.text = text
        self.rhetoric_type = rhetoric_type
        self.category = category
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence = 1.0  # مستوى الثقة في التحديد
        self.explanation = ""  # شرح العنصر البلاغي
        self.related_tokens = []  # الرموز المرتبطة بالعنصر البلاغي
    
    def __str__(self) -> str:
        """تمثيل العنصر البلاغي كنص."""
        return f"{self.rhetoric_type.name}: {self.text}"
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل العنصر البلاغي إلى قاموس."""
        return {
            "text": self.text,
            "rhetoric_type": self.rhetoric_type.name,
            "category": self.category.name,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "related_tokens": [token.text for token in self.related_tokens]
        }


class ArabicRhetoricAnalyzer:
    """محلل البلاغة العربية."""
    
    def __init__(self, rhetoric_patterns_file: str = None):
        """
        تهيئة محلل البلاغة.
        
        Args:
            rhetoric_patterns_file: مسار ملف أنماط البلاغة
        """
        self.logger = logging.getLogger('arabic_nlp.rhetoric.rhetoric_analyzer')
        
        # تحميل أنماط البلاغة
        self.rhetoric_patterns = load_arabic_rhetoric_patterns(rhetoric_patterns_file)
        
        # إنشاء محلل النحو
        self.syntax_analyzer = ArabicSyntaxAnalyzer()
        
        # تهيئة قواميس الأنماط البلاغية
        self.simile_patterns = self.rhetoric_patterns.get('simile', [])
        self.metaphor_patterns = self.rhetoric_patterns.get('metaphor', [])
        self.metonymy_patterns = self.rhetoric_patterns.get('metonymy', [])
        self.alliteration_patterns = self.rhetoric_patterns.get('alliteration', [])
        self.antithesis_patterns = self.rhetoric_patterns.get('antithesis', [])
        self.paronomasia_patterns = self.rhetoric_patterns.get('paronomasia', [])
        self.assonance_patterns = self.rhetoric_patterns.get('assonance', [])
        self.hyperbole_patterns = self.rhetoric_patterns.get('hyperbole', [])
        self.personification_patterns = self.rhetoric_patterns.get('personification', [])
        self.interrogation_patterns = self.rhetoric_patterns.get('interrogation', [])
        self.exclamation_patterns = self.rhetoric_patterns.get('exclamation', [])
        self.imperative_patterns = self.rhetoric_patterns.get('imperative', [])
        self.negation_patterns = self.rhetoric_patterns.get('negation', [])
        self.emphasis_patterns = self.rhetoric_patterns.get('emphasis', [])
        
        # تهيئة قواميس الكلمات المفتاحية
        self.simile_keywords = self.rhetoric_patterns.get('simile_keywords', [])
        self.metaphor_keywords = self.rhetoric_patterns.get('metaphor_keywords', [])
        self.metonymy_keywords = self.rhetoric_patterns.get('metonymy_keywords', [])
    
    def analyze(self, text: str) -> List[RhetoricElement]:
        """
        تحليل النص بلاغياً.
        
        Args:
            text: النص المراد تحليله
            
        Returns:
            قائمة العناصر البلاغية
        """
        # تحليل النص نحوياً
        sentences = self.syntax_analyzer.analyze(text)
        
        # تحليل كل جملة بلاغياً
        rhetoric_elements = []
        
        for sentence in sentences:
            # تحليل التشبيهات
            similes = self._find_similes(sentence)
            rhetoric_elements.extend(similes)
            
            # تحليل الاستعارات
            metaphors = self._find_metaphors(sentence)
            rhetoric_elements.extend(metaphors)
            
            # تحليل الكنايات
            metonymies = self._find_metonymies(sentence)
            rhetoric_elements.extend(metonymies)
            
            # تحليل الجناس
            alliterations = self._find_alliterations(sentence)
            rhetoric_elements.extend(alliterations)
            
            # تحليل الطباق
            antitheses = self._find_antitheses(sentence)
            rhetoric_elements.extend(antitheses)
            
            # تحليل السجع
            assonances = self._find_assonances(sentence)
            rhetoric_elements.extend(assonances)
            
            # تحليل المبالغة
            hyperboles = self._find_hyperboles(sentence)
            rhetoric_elements.extend(hyperboles)
            
            # تحليل التشخيص
            personifications = self._find_personifications(sentence)
            rhetoric_elements.extend(personifications)
            
            # تحليل الاستفهام
            interrogations = self._find_interrogations(sentence)
            rhetoric_elements.extend(interrogations)
            
            # تحليل التعجب
            exclamations = self._find_exclamations(sentence)
            rhetoric_elements.extend(exclamations)
            
            # تحليل الأمر
            imperatives = self._find_imperatives(sentence)
            rhetoric_elements.extend(imperatives)
            
            # تحليل النفي
            negations = self._find_negations(sentence)
            rhetoric_elements.extend(negations)
            
            # تحليل التوكيد
            emphases = self._find_emphases(sentence)
            rhetoric_elements.extend(emphases)
        
        return rhetoric_elements
    
    def _find_similes(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد التشبيهات في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة التشبيهات
        """
        similes = []
        text = sentence.text
        
        # البحث عن أنماط التشبيه
        for pattern in self.simile_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                simile_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للتشبيه
                simile = RhetoricElement(
                    text=simile_text,
                    rhetoric_type=RhetoricType.SIMILE,
                    category=RhetoricCategory.BAYAN,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالتشبيه
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        simile.related_tokens.append(token)
                
                # إضافة شرح للتشبيه
                simile.explanation = "تشبيه: يشبه شيء بشيء آخر باستخدام أداة تشبيه مثل (كـ، مثل، كأن)."
                
                similes.append(simile)
        
        # البحث عن الكلمات المفتاحية للتشبيه
        for keyword in self.simile_keywords:
            if keyword in text:
                # تحديد موقع الكلمة المفتاحية
                start_pos = text.find(keyword)
                end_pos = start_pos + len(keyword)
                
                # تحديد سياق التشبيه (الجملة التي تحتوي على الكلمة المفتاحية)
                context_start = max(0, start_pos - 20)
                context_end = min(len(text), end_pos + 20)
                simile_text = text[context_start:context_end]
                
                # إنشاء عنصر بلاغي للتشبيه
                simile = RhetoricElement(
                    text=simile_text,
                    rhetoric_type=RhetoricType.SIMILE,
                    category=RhetoricCategory.BAYAN,
                    start_pos=context_start,
                    end_pos=context_end
                )
                
                # تحديد الرموز المرتبطة بالتشبيه
                for token in sentence.tokens:
                    if token.position >= context_start and token.position < context_end:
                        simile.related_tokens.append(token)
                
                # إضافة شرح للتشبيه
                simile.explanation = f"تشبيه: يحتوي على كلمة مفتاحية للتشبيه ({keyword})."
                
                # تقليل مستوى الثقة لأن الكلمة المفتاحية قد تكون جزءًا من سياق آخر
                simile.confidence = 0.8
                
                similes.append(simile)
        
        return similes
    
    def _find_metaphors(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد الاستعارات في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة الاستعارات
        """
        metaphors = []
        text = sentence.text
        
        # البحث عن أنماط الاستعارة
        for pattern in self.metaphor_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                metaphor_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للاستعارة
                metaphor = RhetoricElement(
                    text=metaphor_text,
                    rhetoric_type=RhetoricType.METAPHOR,
                    category=RhetoricCategory.BAYAN,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالاستعارة
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        metaphor.related_tokens.append(token)
                
                # إضافة شرح للاستعارة
                metaphor.explanation = "استعارة: تشبيه حذف أحد طرفيه (المشبه أو المشبه به)."
                
                metaphors.append(metaphor)
        
        # البحث عن الكلمات المفتاحية للاستعارة
        for keyword in self.metaphor_keywords:
            if keyword in text:
                # تحديد موقع الكلمة المفتاحية
                start_pos = text.find(keyword)
                end_pos = start_pos + len(keyword)
                
                # تحديد سياق الاستعارة (الجملة التي تحتوي على الكلمة المفتاحية)
                context_start = max(0, start_pos - 20)
                context_end = min(len(text), end_pos + 20)
                metaphor_text = text[context_start:context_end]
                
                # إنشاء عنصر بلاغي للاستعارة
                metaphor = RhetoricElement(
                    text=metaphor_text,
                    rhetoric_type=RhetoricType.METAPHOR,
                    category=RhetoricCategory.BAYAN,
                    start_pos=context_start,
                    end_pos=context_end
                )
                
                # تحديد الرموز المرتبطة بالاستعارة
                for token in sentence.tokens:
                    if token.position >= context_start and token.position < context_end:
                        metaphor.related_tokens.append(token)
                
                # إضافة شرح للاستعارة
                metaphor.explanation = f"استعارة محتملة: تحتوي على كلمة مفتاحية للاستعارة ({keyword})."
                
                # تقليل مستوى الثقة لأن الكلمة المفتاحية قد تكون جزءًا من سياق آخر
                metaphor.confidence = 0.7
                
                metaphors.append(metaphor)
        
        return metaphors
    
    def _find_metonymies(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد الكنايات في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة الكنايات
        """
        metonymies = []
        text = sentence.text
        
        # البحث عن أنماط الكناية
        for pattern in self.metonymy_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                metonymy_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للكناية
                metonymy = RhetoricElement(
                    text=metonymy_text,
                    rhetoric_type=RhetoricType.METONYMY,
                    category=RhetoricCategory.BAYAN,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالكناية
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        metonymy.related_tokens.append(token)
                
                # إضافة شرح للكناية
                metonymy.explanation = "كناية: لفظ أطلق وأريد به لازم معناه مع جواز إرادة المعنى الأصلي."
                
                metonymies.append(metonymy)
        
        # البحث عن الكلمات المفتاحية للكناية
        for keyword in self.metonymy_keywords:
            if keyword in text:
                # تحديد موقع الكلمة المفتاحية
                start_pos = text.find(keyword)
                end_pos = start_pos + len(keyword)
                
                # تحديد سياق الكناية (الجملة التي تحتوي على الكلمة المفتاحية)
                context_start = max(0, start_pos - 20)
                context_end = min(len(text), end_pos + 20)
                metonymy_text = text[context_start:context_end]
                
                # إنشاء عنصر بلاغي للكناية
                metonymy = RhetoricElement(
                    text=metonymy_text,
                    rhetoric_type=RhetoricType.METONYMY,
                    category=RhetoricCategory.BAYAN,
                    start_pos=context_start,
                    end_pos=context_end
                )
                
                # تحديد الرموز المرتبطة بالكناية
                for token in sentence.tokens:
                    if token.position >= context_start and token.position < context_end:
                        metonymy.related_tokens.append(token)
                
                # إضافة شرح للكناية
                metonymy.explanation = f"كناية محتملة: تحتوي على كلمة مفتاحية للكناية ({keyword})."
                
                # تقليل مستوى الثقة لأن الكلمة المفتاحية قد تكون جزءًا من سياق آخر
                metonymy.confidence = 0.6
                
                metonymies.append(metonymy)
        
        return metonymies
    
    def _find_alliterations(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد الجناس في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة الجناس
        """
        alliterations = []
        tokens = sentence.tokens
        
        # البحث عن الجناس بين الكلمات
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                token1 = tokens[i]
                token2 = tokens[j]
                
                # حساب التشابه بين الكلمتين
                similarity = self._calculate_word_similarity(token1.normalized_text, token2.normalized_text)
                
                # إذا كان التشابه كبيرًا، فقد يكون جناسًا
                if similarity >= 0.7 and token1.normalized_text != token2.normalized_text:
                    # تحديد موقع الجناس
                    start_pos = min(token1.position, token2.position)
                    end_pos = max(token1.position, token2.position) + len(token2.text)
                    
                    # تحديد نص الجناس
                    alliteration_text = sentence.text[start_pos:end_pos]
                    
                    # إنشاء عنصر بلاغي للجناس
                    alliteration = RhetoricElement(
                        text=alliteration_text,
                        rhetoric_type=RhetoricType.ALLITERATION,
                        category=RhetoricCategory.BADI,
                        start_pos=start_pos,
                        end_pos=end_pos
                    )
                    
                    # تحديد الرموز المرتبطة بالجناس
                    alliteration.related_tokens = [token1, token2]
                    
                    # إضافة شرح للجناس
                    alliteration.explanation = f"جناس: تشابه في اللفظ واختلاف في المعنى بين الكلمتين ({token1.text} و {token2.text})."
                    
                    # تحديد مستوى الثقة بناءً على التشابه
                    alliteration.confidence = similarity
                    
                    alliterations.append(alliteration)
        
        return alliterations
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        حساب التشابه بين كلمتين.
        
        Args:
            word1: الكلمة الأولى
            word2: الكلمة الثانية
            
        Returns:
            درجة التشابه (0-1)
        """
        # إذا كانت الكلمتان متطابقتين
        if word1 == word2:
            return 1.0
        
        # إذا كانت إحدى الكلمتين فارغة
        if not word1 or not word2:
            return 0.0
        
        # حساب عدد الحروف المشتركة
        common_chars = set(word1) & set(word2)
        
        # حساب طول أطول كلمة
        max_length = max(len(word1), len(word2))
        
        # حساب التشابه
        similarity = len(common_chars) / max_length
        
        # إذا كانت الكلمتان متشابهتين في الترتيب
        if word1[0] == word2[0] and word1[-1] == word2[-1]:
            similarity += 0.2
        
        # تقييد التشابه بين 0 و 1
        return min(1.0, similarity)
    
    def _find_antitheses(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد الطباق في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة الطباق
        """
        antitheses = []
        text = sentence.text
        
        # البحث عن أنماط الطباق
        for pattern in self.antithesis_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                antithesis_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للطباق
                antithesis = RhetoricElement(
                    text=antithesis_text,
                    rhetoric_type=RhetoricType.ANTITHESIS,
                    category=RhetoricCategory.BADI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالطباق
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        antithesis.related_tokens.append(token)
                
                # إضافة شرح للطباق
                antithesis.explanation = "طباق: الجمع بين الشيء وضده في الكلام."
                
                antitheses.append(antithesis)
        
        return antitheses
    
    def _find_assonances(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد السجع في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة السجع
        """
        assonances = []
        text = sentence.text
        
        # البحث عن أنماط السجع
        for pattern in self.assonance_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                assonance_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للسجع
                assonance = RhetoricElement(
                    text=assonance_text,
                    rhetoric_type=RhetoricType.ASSONANCE,
                    category=RhetoricCategory.BADI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالسجع
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        assonance.related_tokens.append(token)
                
                # إضافة شرح للسجع
                assonance.explanation = "سجع: توافق الفاصلتين في الحرف الأخير."
                
                assonances.append(assonance)
        
        return assonances
    
    def _find_hyperboles(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد المبالغة في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة المبالغة
        """
        hyperboles = []
        text = sentence.text
        
        # البحث عن أنماط المبالغة
        for pattern in self.hyperbole_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                hyperbole_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للمبالغة
                hyperbole = RhetoricElement(
                    text=hyperbole_text,
                    rhetoric_type=RhetoricType.HYPERBOLE,
                    category=RhetoricCategory.MAANI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالمبالغة
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        hyperbole.related_tokens.append(token)
                
                # إضافة شرح للمبالغة
                hyperbole.explanation = "مبالغة: وصف شيء بما يتجاوز حده المعتاد."
                
                hyperboles.append(hyperbole)
        
        return hyperboles
    
    def _find_personifications(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد التشخيص في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة التشخيص
        """
        personifications = []
        text = sentence.text
        
        # البحث عن أنماط التشخيص
        for pattern in self.personification_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                personification_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للتشخيص
                personification = RhetoricElement(
                    text=personification_text,
                    rhetoric_type=RhetoricType.PERSONIFICATION,
                    category=RhetoricCategory.BAYAN,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالتشخيص
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        personification.related_tokens.append(token)
                
                # إضافة شرح للتشخيص
                personification.explanation = "تشخيص: إضفاء صفات إنسانية على غير الإنسان."
                
                personifications.append(personification)
        
        return personifications
    
    def _find_interrogations(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد الاستفهام في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة الاستفهام
        """
        interrogations = []
        text = sentence.text
        
        # البحث عن أنماط الاستفهام
        for pattern in self.interrogation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                interrogation_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للاستفهام
                interrogation = RhetoricElement(
                    text=interrogation_text,
                    rhetoric_type=RhetoricType.INTERROGATION,
                    category=RhetoricCategory.MAANI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالاستفهام
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        interrogation.related_tokens.append(token)
                
                # إضافة شرح للاستفهام
                interrogation.explanation = "استفهام: طلب العلم بشيء لم يكن معلومًا من قبل."
                
                interrogations.append(interrogation)
        
        # البحث عن علامات الاستفهام
        if '؟' in text:
            # تحديد موقع علامة الاستفهام
            start_pos = text.find('؟')
            
            # تحديد بداية الجملة الاستفهامية
            context_start = 0
            for i in range(start_pos - 1, -1, -1):
                if text[i] in ['.', '!', '؟', '\n']:
                    context_start = i + 1
                    break
            
            # تحديد نص الاستفهام
            interrogation_text = text[context_start:start_pos + 1]
            
            # إنشاء عنصر بلاغي للاستفهام
            interrogation = RhetoricElement(
                text=interrogation_text,
                rhetoric_type=RhetoricType.INTERROGATION,
                category=RhetoricCategory.MAANI,
                start_pos=context_start,
                end_pos=start_pos + 1
            )
            
            # تحديد الرموز المرتبطة بالاستفهام
            for token in sentence.tokens:
                if token.position >= context_start and token.position < start_pos + 1:
                    interrogation.related_tokens.append(token)
            
            # إضافة شرح للاستفهام
            interrogation.explanation = "استفهام: جملة تنتهي بعلامة استفهام."
            
            interrogations.append(interrogation)
        
        return interrogations
    
    def _find_exclamations(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد التعجب في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة التعجب
        """
        exclamations = []
        text = sentence.text
        
        # البحث عن أنماط التعجب
        for pattern in self.exclamation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                exclamation_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للتعجب
                exclamation = RhetoricElement(
                    text=exclamation_text,
                    rhetoric_type=RhetoricType.EXCLAMATION,
                    category=RhetoricCategory.MAANI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالتعجب
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        exclamation.related_tokens.append(token)
                
                # إضافة شرح للتعجب
                exclamation.explanation = "تعجب: استعظام شيء خارج عن نظائره."
                
                exclamations.append(exclamation)
        
        # البحث عن علامات التعجب
        if '!' in text:
            # تحديد موقع علامة التعجب
            start_pos = text.find('!')
            
            # تحديد بداية جملة التعجب
            context_start = 0
            for i in range(start_pos - 1, -1, -1):
                if text[i] in ['.', '!', '؟', '\n']:
                    context_start = i + 1
                    break
            
            # تحديد نص التعجب
            exclamation_text = text[context_start:start_pos + 1]
            
            # إنشاء عنصر بلاغي للتعجب
            exclamation = RhetoricElement(
                text=exclamation_text,
                rhetoric_type=RhetoricType.EXCLAMATION,
                category=RhetoricCategory.MAANI,
                start_pos=context_start,
                end_pos=start_pos + 1
            )
            
            # تحديد الرموز المرتبطة بالتعجب
            for token in sentence.tokens:
                if token.position >= context_start and token.position < start_pos + 1:
                    exclamation.related_tokens.append(token)
            
            # إضافة شرح للتعجب
            exclamation.explanation = "تعجب: جملة تنتهي بعلامة تعجب."
            
            exclamations.append(exclamation)
        
        return exclamations
    
    def _find_imperatives(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد الأمر في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة الأمر
        """
        imperatives = []
        text = sentence.text
        
        # البحث عن أنماط الأمر
        for pattern in self.imperative_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                imperative_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للأمر
                imperative = RhetoricElement(
                    text=imperative_text,
                    rhetoric_type=RhetoricType.IMPERATIVE,
                    category=RhetoricCategory.MAANI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالأمر
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        imperative.related_tokens.append(token)
                
                # إضافة شرح للأمر
                imperative.explanation = "أمر: طلب الفعل على وجه الاستعلاء."
                
                imperatives.append(imperative)
        
        return imperatives
    
    def _find_negations(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد النفي في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة النفي
        """
        negations = []
        text = sentence.text
        
        # البحث عن أنماط النفي
        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                negation_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للنفي
                negation = RhetoricElement(
                    text=negation_text,
                    rhetoric_type=RhetoricType.NEGATION,
                    category=RhetoricCategory.MAANI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالنفي
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        negation.related_tokens.append(token)
                
                # إضافة شرح للنفي
                negation.explanation = "نفي: إخراج الشيء من حيز الإثبات إلى حيز النفي."
                
                negations.append(negation)
        
        return negations
    
    def _find_emphases(self, sentence: ArabicSentence) -> List[RhetoricElement]:
        """
        تحديد التوكيد في الجملة.
        
        Args:
            sentence: الجملة المراد تحليلها
            
        Returns:
            قائمة التوكيد
        """
        emphases = []
        text = sentence.text
        
        # البحث عن أنماط التوكيد
        for pattern in self.emphasis_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                emphasis_text = text[start_pos:end_pos]
                
                # إنشاء عنصر بلاغي للتوكيد
                emphasis = RhetoricElement(
                    text=emphasis_text,
                    rhetoric_type=RhetoricType.EMPHASIS,
                    category=RhetoricCategory.MAANI,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # تحديد الرموز المرتبطة بالتوكيد
                for token in sentence.tokens:
                    if token.position >= start_pos and token.position < end_pos:
                        emphasis.related_tokens.append(token)
                
                # إضافة شرح للتوكيد
                emphasis.explanation = "توكيد: تقوية المعنى في نفس السامع وإزالة الشك عنه."
                
                emphases.append(emphasis)
        
        return emphases


# --- اختبارات ---
if __name__ == "__main__":
    # إنشاء محلل البلاغة
    analyzer = ArabicRhetoricAnalyzer()
    
    # اختبار تحليل بعض الجمل
    test_sentences = [
        "الولد كالأسد في الشجاعة.",
        "رأيت أسدًا يحمل سلاحًا.",
        "زيد طويل القامة.",
        "هل تستطيع أن تحضر غدًا؟",
        "ما أجمل السماء!",
        "اذهب إلى المدرسة.",
        "لا تتأخر عن الموعد.",
        "إن الله على كل شيء قدير."
    ]
    
    for test_sentence in test_sentences:
        print(f"\nتحليل الجملة: {test_sentence}")
        rhetoric_elements = analyzer.analyze(test_sentence)
        
        if rhetoric_elements:
            print("العناصر البلاغية:")
            for element in rhetoric_elements:
                print(f"  - {element.rhetoric_type.name}: {element.text}")
                print(f"    الفئة: {element.category.name}")
                print(f"    الشرح: {element.explanation}")
                print(f"    مستوى الثقة: {element.confidence:.2f}")
        else:
            print("لم يتم العثور على عناصر بلاغية.")
