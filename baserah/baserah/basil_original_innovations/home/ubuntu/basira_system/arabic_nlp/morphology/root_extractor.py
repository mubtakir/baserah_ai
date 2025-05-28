#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مستخرج جذور الكلمات العربية

هذا الملف يحتوي على فئة مستخرج جذور الكلمات العربية، وهي نسخة محسنة ومعاد هيكلتها
من الكود الأصلي، مع إضافة دعم للأوزان الصرفية وتحسين معالجة الإعلال والإبدال.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import re
import os
import json
import logging
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
import sys

# إضافة المسار إلى حزمة utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import normalize_arabic_text
from utils.data_loader import load_arabic_roots, load_arabic_patterns, load_arabic_affixes

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arabic_nlp.morphology.root_extractor')


class RootExtractionStrategy(Enum):
    """استراتيجيات استخراج الجذور."""
    RULE_BASED = auto()  # استراتيجية قائمة على القواعد
    PATTERN_BASED = auto()  # استراتيجية قائمة على الأنماط
    STATISTICAL = auto()  # استراتيجية إحصائية
    HYBRID = auto()  # استراتيجية هجينة


class ArabicRootExtractorBase(ABC):
    """الفئة الأساسية لمستخرج جذور الكلمات العربية."""
    
    def __init__(self):
        """تهيئة المستخرج."""
        self.logger = logging.getLogger('arabic_nlp.morphology.root_extractor.base')
    
    @abstractmethod
    def extract_root(self, word: str) -> str:
        """
        استخراج جذر الكلمة.
        
        Args:
            word: الكلمة المراد استخراج جذرها
            
        Returns:
            جذر الكلمة
        """
        pass


class RuleBasedRootExtractor(ArabicRootExtractorBase):
    """مستخرج جذور الكلمات العربية القائم على القواعد."""
    
    def __init__(self, 
                 roots_file: str = None, 
                 patterns_file: str = None, 
                 affixes_file: str = None):
        """
        تهيئة المستخرج.
        
        Args:
            roots_file: مسار ملف الجذور
            patterns_file: مسار ملف الأنماط
            affixes_file: مسار ملف الزوائد
        """
        super().__init__()
        self.logger = logging.getLogger('arabic_nlp.morphology.root_extractor.rule_based')
        
        # تحميل البيانات
        self.known_roots = load_arabic_roots(roots_file)
        self.patterns = load_arabic_patterns(patterns_file)
        affixes_data = load_arabic_affixes(affixes_file)
        
        # استخراج السوابق واللواحق وأدوات التعريف
        self.definite_articles = affixes_data.get('definite_articles', [])
        self.prefixes = affixes_data.get('prefixes', [])
        self.suffixes = affixes_data.get('suffixes', [])
        
        # تأكد من ترتيب السوابق واللواحق حسب الطول
        self.definite_articles = sorted(self.definite_articles, key=len, reverse=True)
        self.prefixes = sorted(self.prefixes, key=len, reverse=True)
        self.suffixes = sorted(self.suffixes, key=len, reverse=True)
        
        # تكوين أنماط التعبيرات النمطية
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652]')  # فتحة، ضمة، كسرة، سكون، تنوين...
        self.tatweel_pattern = re.compile(r'\u0640')  # تطويل ـ
        
        # تكوين خريطة التطبيع
        self.arabic_normalization_map = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # توحيد الألفات
            'ى': 'ي',                      # توحيد الياء
            'ة': 'ه',                      # توحيد التاء المربوطة
        }
        
        # تكوين الثوابت
        self.min_stem_length = 2  # أقل طول للجذع بعد إزالة الزوائد
        self.min_root_length = 2  # أقل طول للجذر
        self.typical_root_length = 3  # الطول النموذجي للجذر
    
    def normalize(self, word: str) -> str:
        """
        تطبيع الكلمة.
        
        Args:
            word: الكلمة المراد تطبيعها
            
        Returns:
            الكلمة المطبعة
        """
        return normalize_arabic_text(
            word,
            remove_diacritics=True,
            remove_tatweel=True,
            normalize_alef=True,
            normalize_yeh=True,
            normalize_teh_marbuta=True
        )
    
    def strip_affixes(self, word: str, min_len: int = 2) -> str:
        """
        إزالة السوابق واللواحق.
        
        Args:
            word: الكلمة المراد إزالة الزوائد منها
            min_len: أقل طول للجذع بعد إزالة الزوائد
            
        Returns:
            الكلمة بعد إزالة الزوائد
        """
        original_word = word
        current_word = word
        
        # التعامل الأولي مع "ال" التعريف ومشتقاتها
        for da_prefix in self.definite_articles:
            if current_word.startswith(da_prefix) and len(current_word) - len(da_prefix) >= min_len:
                current_word = current_word[len(da_prefix):]
                break  # نزيل واحدة فقط ونكمل
        
        # حلقة لإزالة السوابق واللواحق بالتناوب أو بشكل متكرر
        can_reduce = True
        while can_reduce:
            can_reduce = False
            # محاولة إزالة السوابق
            for p in self.prefixes:
                if current_word.startswith(p) and len(current_word) - len(p) >= min_len:
                    current_word = current_word[len(p):]
                    can_reduce = True
                    break
            if can_reduce:
                continue  # إذا أزيل شيء، أعد المحاولة من البداية
            
            # محاولة إزالة اللواحق
            for s in self.suffixes:
                if current_word.endswith(s) and len(current_word) - len(s) >= min_len:
                    current_word = current_word[:-len(s)]
                    can_reduce = True
                    break
        
        if not current_word:  # إذا أُفرغت الكلمة تماماً
            return original_word  # أرجع الكلمة الأصلية (بعد التطبيع)
        
        return current_word
    
    def apply_patterns(self, stem: str) -> str:
        """
        تطبيق الأنماط الصرفية.
        
        Args:
            stem: الجذع المراد تطبيق الأنماط عليه
            
        Returns:
            الجذر المستخرج
        """
        # إذا كان الجذع بالفعل في قائمة الجذور المعروفة بالطول المناسب
        if (len(stem) == 2 or len(stem) == 3 or len(stem) == 4) and stem in self.known_roots:
            return stem
        
        # تطبيق الأنماط المعروفة
        for pattern_name, pattern_info in self.patterns.items():
            pattern = pattern_info.get('pattern', '')
            root_indices = pattern_info.get('root_indices', [])
            
            # تحقق من تطابق الجذع مع النمط
            if self._match_pattern(stem, pattern, root_indices):
                # استخراج الجذر من الجذع باستخدام مؤشرات الجذر
                root = self._extract_root_from_stem(stem, pattern, root_indices)
                if root and len(root) >= self.min_root_length:
                    return root
        
        # أنماط بسيطة إضافية
        # نمط فاعل -> فعل (مثل كاتب -> كتب)
        if len(stem) == 4 and stem[1] == 'ا':
            candidate = stem[0] + stem[2] + stem[3]
            if len(candidate) == 3:
                return candidate
        
        # نمط استفعل -> فعل
        if len(stem) == 6 and stem.startswith("است"):
            candidate = stem[3:]
            if len(candidate) == 3:
                return candidate
        
        # نمط افتعل -> فعل (اجتمع -> جمع)
        if len(stem) == 5 and stem.startswith("ا") and stem[2] == "ت":
            candidate = stem[1] + stem[3] + stem[4]
            if len(candidate) == 3:
                return candidate
        
        # نمط انفعل -> فعل (انكسر -> كسر)
        if len(stem) == 5 and stem.startswith("ان"):
            candidate = stem[2:]
            if len(candidate) == 3:
                return candidate
        
        # نمط تفعّل -> فعل (تقدّم -> قدم)
        if len(stem) == 4 and stem[1] == stem[2]:
            candidate = stem[0] + stem[1] + stem[3]
            if len(candidate) == 3:
                return candidate
        
        # نمط تفعْلَل -> فعلل (تزلزل -> زلزل)
        if len(stem) == 5 and stem.startswith("ت") and stem[2] == stem[4]:
            candidate = stem[1:]
            if len(candidate) == 4:
                return candidate
        
        # إذا كان الجذع ثلاثي أو رباعي بعد كل هذا، فهو مرشح جيد
        if len(stem) == 3 or len(stem) == 4:
            return stem
        
        return stem
    
    def _match_pattern(self, stem: str, pattern: str, root_indices: List[int]) -> bool:
        """
        تحقق من تطابق الجذع مع النمط.
        
        Args:
            stem: الجذع
            pattern: النمط
            root_indices: مؤشرات الجذر في النمط
            
        Returns:
            True إذا كان الجذع يطابق النمط، وإلا False
        """
        # تحقق من الطول
        if len(stem) != len(pattern):
            return False
        
        # تحقق من تطابق الحروف غير الجذرية
        for i in range(len(pattern)):
            if i not in root_indices and pattern[i] != stem[i]:
                return False
        
        return True
    
    def _extract_root_from_stem(self, stem: str, pattern: str, root_indices: List[int]) -> str:
        """
        استخراج الجذر من الجذع باستخدام مؤشرات الجذر.
        
        Args:
            stem: الجذع
            pattern: النمط
            root_indices: مؤشرات الجذر في النمط
            
        Returns:
            الجذر المستخرج
        """
        root = ''
        for i in root_indices:
            if i < len(stem):
                root += stem[i]
        
        return root
    
    def handle_weak_letters(self, root_candidate: str) -> str:
        """
        معالجة الحروف المعتلة والتضعيف.
        
        Args:
            root_candidate: مرشح الجذر
            
        Returns:
            الجذر بعد معالجة الحروف المعتلة والتضعيف
        """
        # معالجة الجذور الثلاثية
        if len(root_candidate) == 3:
            # معالجة الهمزة في الوسط (قئل، سئل، بئس)
            if root_candidate[1] == 'ئ':
                # محاولة استبدالها بـ و أو ي إذا كان ذلك يشكل جذرًا معروفًا
                candidate_waw = root_candidate[0] + 'و' + root_candidate[2]
                if candidate_waw in self.known_roots:
                    return candidate_waw
                
                candidate_yaa = root_candidate[0] + 'ي' + root_candidate[2]
                if candidate_yaa in self.known_roots:
                    return candidate_yaa
            
            # إذا كان الحرف الثاني والثالث متماثلين (مدد، شدد)
            if root_candidate[1] == root_candidate[2]:
                # بعض المناهج تعتبر الجذر مدد وبعضها مد
                # إذا "مد" في قائمة الجذور، نفضلها
                simplified = root_candidate[0] + root_candidate[1]
                if simplified in self.known_roots:
                    return simplified
        
        # إذا كان طوله 4 والحرفان الأوسطان مكرران (مثل قططع من قطّع)
        if len(root_candidate) == 4 and root_candidate[1] == root_candidate[2]:
            simplified = root_candidate[0] + root_candidate[1] + root_candidate[3]
            if len(simplified) == 3:  # مثل قططع -> قطع
                return simplified
        
        return root_candidate
    
    def extract_root(self, word: str) -> str:
        """
        استخراج جذر الكلمة.
        
        Args:
            word: الكلمة المراد استخراج جذرها
            
        Returns:
            جذر الكلمة
        """
        if not word or not isinstance(word, str) or not word.strip():
            return ""  # كلمة فارغة أو غير نصية
        
        # 1. تطبيع الكلمة
        normalized_word = self.normalize(word)
        if not normalized_word:
            return ""  # إذا نتج عن التطبيع كلمة فارغة
        
        # إذا كانت الكلمة قصيرة جداً (أقل من 3 حروف) فقد تكون هي الجذر
        if len(normalized_word) <= self.min_root_length + 1:  # مثال: 2 أو 3 حروف
            if normalized_word in self.known_roots:  # لو "يد" أو "كتب"
                return normalized_word
            # إذا لم تكن معروفة ولكنها بطول الجذر النموذجي، قد تكون هي الجذر
            if len(normalized_word) == self.typical_root_length:
                return normalized_word
        
        # 2. إزالة السوابق واللواحق بشكل متكرر
        stem = self.strip_affixes(normalized_word, self.min_stem_length)
        
        # إذا أصبح الجذع قصيراً جداً، نتحقق
        if len(stem) < self.min_root_length:
            # ربما الكلمة الأصلية كانت هي الجذر
            if len(normalized_word) >= self.min_root_length and len(normalized_word) <= 4:  # 4 للجذور الرباعية
                if normalized_word in self.known_roots:
                    return normalized_word
                # إذا طولها 3 أو 4 ولم تزل منها شيء كثير، نعتبرها الجذر
                if normalized_word == stem or len(normalized_word) - len(stem) <= 1:
                    return normalized_word
            return stem  # أو أرجع الجذع القصير جداً (قد يكون صحيحاً كـ "يد")
        
        # 3. تطبيق الأنماط الصرفية
        pattern_applied_stem = self.apply_patterns(stem)
        
        # 4. محاولة معالجة الحروف المعتلة والتضعيف
        final_root_candidate = self.handle_weak_letters(pattern_applied_stem)
        
        # 5. التحقق النهائي من الطول والجذور المعروفة
        if len(final_root_candidate) < self.min_root_length:
            # إذا أدت المعالجات إلى جذر قصير جداً، قد نعود إلى ما قبلها
            if len(stem) >= self.min_root_length and len(stem) <= 4:
                if stem in self.known_roots:
                    return stem
                return stem  # أو pattern_applied_stem إذا كان أطول
            return final_root_candidate  # أو نتركها كما هي
        
        # تفضيل الجذور المعروفة
        if final_root_candidate in self.known_roots:
            return final_root_candidate
        
        # إذا لم يكن معروفاً، ولكن طوله مناسب (3 أو 4)، فهو مرشح جيد
        if len(final_root_candidate) == 3 or len(final_root_candidate) == 4:
            return final_root_candidate
        
        # إذا كان أطول من 4، قد يكون خطأ أو جذر خماسي (نادر) أو لم تتم معالجته جيداً
        if len(final_root_candidate) > 4:
            # محاولة أخيرة لتقصيره إذا كان يبدأ بزيادات شائعة لم يتم التقاطها
            if final_root_candidate.startswith("ا") and len(final_root_candidate[1:]) == 4:  # مثل اخضوضر -> خضضر
                return final_root_candidate[1:]
            return final_root_candidate[:4]  # كحل أخير، نأخذ أول 4 أحرف
        
        # إذا كان طوله 2 وليس في القائمة المعروفة، لا يزال احتمالاً
        if len(final_root_candidate) == 2:
            return final_root_candidate
        
        # كحل أخير تماماً، إذا فشل كل شيء، أرجع الجذع الأولي بعد إزالة الزوائد
        if len(stem) >= self.min_root_length and len(stem) <= 4:
            return stem
        
        return final_root_candidate


class PatternBasedRootExtractor(ArabicRootExtractorBase):
    """مستخرج جذور الكلمات العربية القائم على الأنماط."""
    
    def __init__(self, patterns_file: str = None, roots_file: str = None):
        """
        تهيئة المستخرج.
        
        Args:
            patterns_file: مسار ملف الأنماط
            roots_file: مسار ملف الجذور
        """
        super().__init__()
        self.logger = logging.getLogger('arabic_nlp.morphology.root_extractor.pattern_based')
        
        # تحميل البيانات
        self.patterns = load_arabic_patterns(patterns_file)
        self.known_roots = load_arabic_roots(roots_file)
    
    def extract_root(self, word: str) -> str:
        """
        استخراج جذر الكلمة.
        
        Args:
            word: الكلمة المراد استخراج جذرها
            
        Returns:
            جذر الكلمة
        """
        # تطبيع الكلمة
        normalized_word = normalize_arabic_text(
            word,
            remove_diacritics=True,
            remove_tatweel=True,
            normalize_alef=True,
            normalize_yeh=True,
            normalize_teh_marbuta=True
        )
        
        # تطبيق الأنماط
        for pattern_name, pattern_info in self.patterns.items():
            pattern = pattern_info.get('pattern', '')
            root_indices = pattern_info.get('root_indices', [])
            
            # تحقق من تطابق الكلمة مع النمط
            if len(normalized_word) == len(pattern):
                # استخراج الجذر المحتمل
                root_candidate = ''
                for i in root_indices:
                    if i < len(normalized_word):
                        root_candidate += normalized_word[i]
                
                # التحقق من الجذر
                if root_candidate in self.known_roots:
                    return root_candidate
        
        # إذا لم يتم العثور على جذر، أرجع الكلمة المطبعة
        return normalized_word


class HybridRootExtractor(ArabicRootExtractorBase):
    """مستخرج جذور الكلمات العربية الهجين."""
    
    def __init__(self, 
                 roots_file: str = None, 
                 patterns_file: str = None, 
                 affixes_file: str = None):
        """
        تهيئة المستخرج.
        
        Args:
            roots_file: مسار ملف الجذور
            patterns_file: مسار ملف الأنماط
            affixes_file: مسار ملف الزوائد
        """
        super().__init__()
        self.logger = logging.getLogger('arabic_nlp.morphology.root_extractor.hybrid')
        
        # إنشاء المستخرجات
        self.rule_based_extractor = RuleBasedRootExtractor(
            roots_file=roots_file,
            patterns_file=patterns_file,
            affixes_file=affixes_file
        )
        
        self.pattern_based_extractor = PatternBasedRootExtractor(
            patterns_file=patterns_file,
            roots_file=roots_file
        )
    
    def extract_root(self, word: str) -> str:
        """
        استخراج جذر الكلمة.
        
        Args:
            word: الكلمة المراد استخراج جذرها
            
        Returns:
            جذر الكلمة
        """
        # تطبيع الكلمة
        normalized_word = normalize_arabic_text(
            word,
            remove_diacritics=True,
            remove_tatweel=True,
            normalize_alef=True,
            normalize_yeh=True,
            normalize_teh_marbuta=True
        )
        
        # استخراج الجذر باستخدام المستخرج القائم على القواعد
        rule_based_root = self.rule_based_extractor.extract_root(normalized_word)
        
        # استخراج الجذر باستخدام المستخرج القائم على الأنماط
        pattern_based_root = self.pattern_based_extractor.extract_root(normalized_word)
        
        # اختيار الجذر الأفضل
        if rule_based_root in self.rule_based_extractor.known_roots:
            return rule_based_root
        
        if pattern_based_root in self.rule_based_extractor.known_roots:
            return pattern_based_root
        
        # إذا كان طول الجذر القائم على القواعد 3 أو 4، فهو مرشح جيد
        if len(rule_based_root) == 3 or len(rule_based_root) == 4:
            return rule_based_root
        
        # إذا كان طول الجذر القائم على الأنماط 3 أو 4، فهو مرشح جيد
        if len(pattern_based_root) == 3 or len(pattern_based_root) == 4:
            return pattern_based_root
        
        # إذا فشل كل شيء، أرجع الجذر القائم على القواعد
        return rule_based_root


class ArabicRootExtractor:
    """واجهة موحدة لمستخرج جذور الكلمات العربية."""
    
    def __init__(self, 
                 strategy: RootExtractionStrategy = RootExtractionStrategy.HYBRID,
                 roots_file: str = None, 
                 patterns_file: str = None, 
                 affixes_file: str = None):
        """
        تهيئة المستخرج.
        
        Args:
            strategy: استراتيجية استخراج الجذور
            roots_file: مسار ملف الجذور
            patterns_file: مسار ملف الأنماط
            affixes_file: مسار ملف الزوائد
        """
        self.logger = logging.getLogger('arabic_nlp.morphology.root_extractor')
        
        # تحديد الاستراتيجية
        self.strategy = strategy
        
        # إنشاء المستخرج المناسب
        if strategy == RootExtractionStrategy.RULE_BASED:
            self.extractor = RuleBasedRootExtractor(
                roots_file=roots_file,
                patterns_file=patterns_file,
                affixes_file=affixes_file
            )
        elif strategy == RootExtractionStrategy.PATTERN_BASED:
            self.extractor = PatternBasedRootExtractor(
                patterns_file=patterns_file,
                roots_file=roots_file
            )
        elif strategy == RootExtractionStrategy.HYBRID:
            self.extractor = HybridRootExtractor(
                roots_file=roots_file,
                patterns_file=patterns_file,
                affixes_file=affixes_file
            )
        else:
            # الاستراتيجية الافتراضية هي الهجينة
            self.extractor = HybridRootExtractor(
                roots_file=roots_file,
                patterns_file=patterns_file,
                affixes_file=affixes_file
            )
    
    def get_root(self, word: str) -> str:
        """
        استخراج جذر الكلمة.
        
        Args:
            word: الكلمة المراد استخراج جذرها
            
        Returns:
            جذر الكلمة
        """
        return self.extractor.extract_root(word)


# --- اختبارات ---
if __name__ == "__main__":
    # إنشاء مستخرج الجذور
    extractor = ArabicRootExtractor(
        strategy=RootExtractionStrategy.HYBRID
    )
    
    # اختبار بعض الكلمات
    test_cases = {
        "مَدْرَسَةٌ": "درس",  # مع حركات وتاء مربوطة
        "يَسْتَخْرِجُونَ": "خرج",  # مضارع جمع
        "مُعَلِّمِينَ": "علم",  # جمع مذكر سالم
        "مُسْتَشْفَيَاتٍ": "شفي",  # جمع مؤنث سالم
        "فَاعِل": "فعل",  # اسم فاعل
        "مُتَفَاعِل": "فعل",  # اسم فاعل من تفاعل
        "كِتَاب": "كتب",  # مصدر أو اسم
        "كَاتِب": "كتب",  # اسم فاعل
        "انْتَظَرَ": "نظر",  # فعل افتعل (مع قلب واو إلى تاء)
        "اسْتَغْفَرَ": "غفر",  # فعل استفعل
        "الْمَكْتَبَةُ": "كتب",  # مع ال التعريف
        "بِالْقَلَمِ": "قلم",  # مع حرف جر وال التعريف
        "وَالْكَاتِبَاتُ": "كتب",  # مع واو العطف وال التعريف وجمع مؤنث
        "تَزَلْزَلَ": "زلزل",  # فعل تفعلل (رباعي مزيد)
        "مُدَحْرِجٌ": "دحرج",  # اسم فاعل من فعلل
        "قَالَ": "قول",  # معتل أجوف
        "بَاعَ": "بيع",  # معتل أجوف
        "رَمَى": "رمي",  # معتل ناقص
        "دَعَا": "دعو",  # معتل ناقص (الياء أصلها واو)
        "شَدَّ": "شدد",  # مضعف ثلاثي
        "يَدٌ": "يد",  # اسم ثنائي
        "أَخَذَ": "اخذ",  # مهموز الفاء
        "سَأَلَ": "سال",  # مهموز العين
        "قَرَأَ": "قرا",  # مهموز اللام
    }
    
    # طباعة النتائج
    print(f"{'الكلمة':<18} | {'الناتج':<6} | {'المتوقع':<6} | {'الحالة'}")
    print("-" * 50)
    
    correct_count = 0
    for word, expected_root in test_cases.items():
        extracted_root = extractor.get_root(word)
        status = "✅" if extracted_root == expected_root else "❌"
        if status == "✅":
            correct_count += 1
        print(f"{word:<18} | {extracted_root:<6} | {expected_root:<6} | {status}")
    
    print("-" * 50)
    accuracy = (correct_count / len(test_cases)) * 100
    print(f"إجمالي الاختبارات: {len(test_cases)}")
    print(f"صحيح: {correct_count}")
    print(f"الدقة: {accuracy:.2f}%")
