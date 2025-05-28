#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة البحث الذكي لنظام بصيرة

هذا الملف يحدد وحدة البحث الذكي لنظام بصيرة، التي تمكن النظام من البحث
في الإنترنت واستخراج المعلومات ذات الصلة بطريقة ذكية ومتكاملة مع النظام المعرفي.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
import time
import random
import re
import requests
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import concurrent.futures

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import SemanticVector, ConceptualGraph, ConceptNode, ConceptRelation
from knowledge_extraction_generation import KnowledgeExtractor, KnowledgeGenerator

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_intelligent_search')


class SearchMode(Enum):
    """أنماط البحث."""
    GENERAL = auto()  # بحث عام
    ACADEMIC = auto()  # بحث أكاديمي
    NEWS = auto()  # بحث في الأخبار
    IMAGES = auto()  # بحث عن صور
    VIDEOS = auto()  # بحث عن فيديوهات
    SPECIALIZED = auto()  # بحث متخصص
    DEEP = auto()  # بحث عميق
    HYBRID = auto()  # بحث هجين


class ContentType(Enum):
    """أنواع المحتوى."""
    TEXT = auto()  # نص
    IMAGE = auto()  # صورة
    VIDEO = auto()  # فيديو
    AUDIO = auto()  # صوت
    DOCUMENT = auto()  # مستند
    DATASET = auto()  # مجموعة بيانات
    CODE = auto()  # كود
    MIXED = auto()  # مختلط


@dataclass
class SearchConfig:
    """تكوين البحث."""
    query: str  # استعلام البحث
    mode: SearchMode = SearchMode.GENERAL  # نمط البحث
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.TEXT])  # أنواع المحتوى المطلوبة
    max_results: int = 10  # أقصى عدد للنتائج
    time_limit: Optional[str] = None  # حد زمني للنتائج (مثل "day", "week", "month", "year")
    language: Optional[str] = None  # لغة النتائج
    region: Optional[str] = None  # منطقة النتائج
    safe_search: bool = True  # تصفية المحتوى غير اللائق
    include_related_concepts: bool = True  # تضمين المفاهيم ذات الصلة
    semantic_vector: Optional[SemanticVector] = None  # المتجه الدلالي للبحث
    additional_params: Dict[str, Any] = field(default_factory=dict)  # معلمات إضافية


@dataclass
class SearchResult:
    """نتيجة بحث فردية."""
    title: str  # عنوان النتيجة
    url: str  # رابط النتيجة
    snippet: str  # مقتطف من النتيجة
    content_type: ContentType  # نوع المحتوى
    source: str  # مصدر النتيجة
    date: Optional[str] = None  # تاريخ النتيجة
    author: Optional[str] = None  # مؤلف النتيجة
    relevance_score: float = 1.0  # درجة الصلة
    semantic_vector: Optional[SemanticVector] = None  # المتجه الدلالي للنتيجة
    metadata: Dict[str, Any] = field(default_factory=dict)  # بيانات وصفية إضافية
    
    def to_dict(self) -> Dict[str, Any]:
        """
        تحويل النتيجة إلى قاموس.
        
        Returns:
            قاموس يمثل النتيجة
        """
        result = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content_type": self.content_type.name,
            "source": self.source,
            "relevance_score": self.relevance_score
        }
        
        if self.date:
            result["date"] = self.date
        
        if self.author:
            result["author"] = self.author
        
        if self.semantic_vector:
            result["semantic_vector"] = self.semantic_vector.to_dict()
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """
        إنشاء نتيجة من قاموس.
        
        Args:
            data: قاموس يمثل النتيجة
            
        Returns:
            نتيجة البحث
        """
        semantic_vector = None
        if "semantic_vector" in data:
            semantic_vector = SemanticVector.from_dict(data["semantic_vector"])
        
        return cls(
            title=data["title"],
            url=data["url"],
            snippet=data["snippet"],
            content_type=ContentType[data["content_type"]],
            source=data["source"],
            date=data.get("date"),
            author=data.get("author"),
            relevance_score=data.get("relevance_score", 1.0),
            semantic_vector=semantic_vector,
            metadata=data.get("metadata", {})
        )


@dataclass
class SearchResponse:
    """استجابة البحث."""
    query: str  # استعلام البحث
    results: List[SearchResult]  # نتائج البحث
    total_results: int  # إجمالي عدد النتائج
    search_time: float  # وقت البحث بالثواني
    mode: SearchMode  # نمط البحث
    related_queries: List[str] = field(default_factory=list)  # استعلامات ذات صلة
    related_concepts: List[str] = field(default_factory=list)  # مفاهيم ذات صلة
    metadata: Dict[str, Any] = field(default_factory=dict)  # بيانات وصفية إضافية
    
    def to_dict(self) -> Dict[str, Any]:
        """
        تحويل الاستجابة إلى قاموس.
        
        Returns:
            قاموس يمثل الاستجابة
        """
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "total_results": self.total_results,
            "search_time": self.search_time,
            "mode": self.mode.name,
            "related_queries": self.related_queries,
            "related_concepts": self.related_concepts,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResponse':
        """
        إنشاء استجابة من قاموس.
        
        Args:
            data: قاموس يمثل الاستجابة
            
        Returns:
            استجابة البحث
        """
        results = [SearchResult.from_dict(result) for result in data["results"]]
        
        return cls(
            query=data["query"],
            results=results,
            total_results=data["total_results"],
            search_time=data["search_time"],
            mode=SearchMode[data["mode"]],
            related_queries=data.get("related_queries", []),
            related_concepts=data.get("related_concepts", []),
            metadata=data.get("metadata", {})
        )


class SearchEngineBase(ABC):
    """الفئة الأساسية لمحرك البحث."""
    
    def __init__(self):
        """تهيئة محرك البحث."""
        self.logger = logging.getLogger('basira_intelligent_search.base')
    
    @abstractmethod
    def search(self, config: SearchConfig) -> SearchResponse:
        """
        إجراء بحث.
        
        Args:
            config: تكوين البحث
            
        Returns:
            استجابة البحث
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        التحقق من توفر محرك البحث.
        
        Returns:
            True إذا كان محرك البحث متوفراً، وإلا False
        """
        pass


class MockSearchEngine(SearchEngineBase):
    """محرك بحث وهمي للاختبار."""
    
    def __init__(self):
        """تهيئة محرك البحث الوهمي."""
        super().__init__()
        self.logger = logging.getLogger('basira_intelligent_search.mock')
    
    def search(self, config: SearchConfig) -> SearchResponse:
        """
        إجراء بحث وهمي.
        
        Args:
            config: تكوين البحث
            
        Returns:
            استجابة البحث
        """
        self.logger.info(f"إجراء بحث وهمي عن: {config.query}")
        
        # قياس وقت البحث
        start_time = time.time()
        
        # إنشاء نتائج وهمية
        results = []
        for i in range(min(config.max_results, 10)):
            result = SearchResult(
                title=f"نتيجة {i+1} لـ {config.query}",
                url=f"https://example.com/result/{i+1}",
                snippet=f"هذا مقتطف وهمي للنتيجة {i+1} من البحث عن {config.query}. يحتوي على معلومات وهمية لأغراض الاختبار.",
                content_type=random.choice(list(ContentType)),
                source="محرك بحث وهمي",
                date=f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                author=f"مؤلف وهمي {i+1}",
                relevance_score=random.uniform(0.5, 1.0),
                metadata={
                    "is_mock": True,
                    "mock_id": i+1
                }
            )
            results.append(result)
        
        # حساب وقت البحث
        search_time = time.time() - start_time
        
        return SearchResponse(
            query=config.query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            mode=config.mode,
            related_queries=[f"{config.query} متقدم", f"{config.query} أساسيات", f"{config.query} أمثلة"],
            related_concepts=["مفهوم وهمي 1", "مفهوم وهمي 2", "مفهوم وهمي 3"],
            metadata={
                "engine": "MockSearchEngine",
                "note": "هذا محرك بحث وهمي للاختبار فقط"
            }
        )
    
    def is_available(self) -> bool:
        """
        التحقق من توفر محرك البحث.
        
        Returns:
            True دائماً لأن محرك البحث الوهمي متوفر دائماً
        """
        return True


class GoogleSearchEngine(SearchEngineBase):
    """محرك بحث Google."""
    
    def __init__(self, api_key: Optional[str] = None, cx: Optional[str] = None):
        """
        تهيئة محرك بحث Google.
        
        Args:
            api_key: مفتاح API لـ Google Custom Search
            cx: معرف محرك البحث المخصص
        """
        super().__init__()
        self.logger = logging.getLogger('basira_intelligent_search.google')
        self.api_key = api_key
        self.cx = cx
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, config: SearchConfig) -> SearchResponse:
        """
        إجراء بحث باستخدام Google.
        
        Args:
            config: تكوين البحث
            
        Returns:
            استجابة البحث
        """
        self.logger.info(f"إجراء بحث Google عن: {config.query}")
        
        # التحقق من توفر المفاتيح
        if not self.api_key or not self.cx:
            self.logger.warning("مفاتيح API غير متوفرة، استخدام محرك البحث الوهمي كبديل")
            return MockSearchEngine().search(config)
        
        # قياس وقت البحث
        start_time = time.time()
        
        try:
            # إعداد معلمات البحث
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": config.query,
                "num": min(config.max_results, 10)  # أقصى عدد للنتائج هو 10 في API المجاني
            }
            
            # إضافة معلمات إضافية
            if config.language:
                params["lr"] = f"lang_{config.language}"
            
            if config.region:
                params["gl"] = config.region
            
            if config.safe_search:
                params["safe"] = "active"
            
            if config.time_limit:
                # تحويل حد الوقت إلى تنسيق Google
                if config.time_limit == "day":
                    params["dateRestrict"] = "d1"
                elif config.time_limit == "week":
                    params["dateRestrict"] = "w1"
                elif config.time_limit == "month":
                    params["dateRestrict"] = "m1"
                elif config.time_limit == "year":
                    params["dateRestrict"] = "y1"
            
            # تحديد نوع البحث
            if ContentType.IMAGE in config.content_types:
                params["searchType"] = "image"
            
            # إجراء طلب البحث
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # معالجة النتائج
            results = []
            for item in data.get("items", []):
                content_type = ContentType.TEXT
                if "image" in item:
                    content_type = ContentType.IMAGE
                
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    content_type=content_type,
                    source="Google",
                    date=None,  # تاريخ غير متوفر في API
                    author=None,  # مؤلف غير متوفر في API
                    relevance_score=1.0,  # درجة الصلة غير متوفرة في API
                    metadata={
                        "display_link": item.get("displayLink", ""),
                        "mime_type": item.get("mime", ""),
                        "file_format": item.get("fileFormat", "")
                    }
                )
                results.append(result)
            
            # استخراج الاستعلامات ذات الصلة
            related_queries = []
            if "queries" in data and "related" in data["queries"]:
                for query in data["queries"]["related"]:
                    related_queries.append(query.get("title", ""))
            
            # حساب وقت البحث
            search_time = time.time() - start_time
            
            return SearchResponse(
                query=config.query,
                results=results,
                total_results=int(data.get("searchInformation", {}).get("totalResults", len(results))),
                search_time=search_time,
                mode=config.mode,
                related_queries=related_queries,
                related_concepts=[],  # المفاهيم ذات الصلة غير متوفرة في API
                metadata={
                    "engine": "GoogleSearchEngine",
                    "search_time_formatted": data.get("searchInformation", {}).get("formattedSearchTime", ""),
                    "formatted_total_results": data.get("searchInformation", {}).get("formattedTotalResults", "")
                }
            )
        
        except Exception as e:
            self.logger.error(f"فشل في إجراء بحث Google: {e}")
            return MockSearchEngine().search(config)
    
    def is_available(self) -> bool:
        """
        التحقق من توفر محرك البحث.
        
        Returns:
            True إذا كان محرك البحث متوفراً، وإلا False
        """
        return self.api_key is not None and self.cx is not None


class WebScrapingEngine(SearchEngineBase):
    """محرك بحث يعتمد على استخراج البيانات من الويب."""
    
    def __init__(self, base_search_engine: Optional[SearchEngineBase] = None):
        """
        تهيئة محرك استخراج البيانات من الويب.
        
        Args:
            base_search_engine: محرك البحث الأساسي للحصول على الروابط
        """
        super().__init__()
        self.logger = logging.getLogger('basira_intelligent_search.web_scraping')
        self.base_search_engine = base_search_engine or MockSearchEngine()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def _scrape_page(self, url: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        استخراج البيانات من صفحة ويب.
        
        Args:
            url: رابط الصفحة
            
        Returns:
            عنوان الصفحة، محتوى الصفحة، بيانات وصفية إضافية
        """
        try:
            # إجراء طلب HTTP
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # تحليل HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # استخراج العنوان
            title = soup.title.string if soup.title else ""
            
            # استخراج المحتوى النصي
            paragraphs = soup.find_all('p')
            content = "\n".join([p.get_text() for p in paragraphs])
            
            # استخراج البيانات الوصفية
            metadata = {}
            
            # استخراج الكلمات المفتاحية
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords:
                metadata["keywords"] = meta_keywords.get("content", "").split(",")
            
            # استخراج الوصف
            meta_description = soup.find("meta", attrs={"name": "description"})
            if meta_description:
                metadata["description"] = meta_description.get("content", "")
            
            # استخراج المؤلف
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author:
                metadata["author"] = meta_author.get("content", "")
            
            # استخراج التاريخ
            meta_date = soup.find("meta", attrs={"name": "date"})
            if meta_date:
                metadata["date"] = meta_date.get("content", "")
            
            return title, content, metadata
        
        except Exception as e:
            self.logger.error(f"فشل في استخراج البيانات من {url}: {e}")
            return "", "", {}
    
    def search(self, config: SearchConfig) -> SearchResponse:
        """
        إجراء بحث باستخدام استخراج البيانات من الويب.
        
        Args:
            config: تكوين البحث
            
        Returns:
            استجابة البحث
        """
        self.logger.info(f"إجراء بحث باستخدام استخراج البيانات من الويب عن: {config.query}")
        
        # قياس وقت البحث
        start_time = time.time()
        
        # الحصول على نتائج البحث الأساسية
        base_response = self.base_search_engine.search(config)
        
        # استخراج البيانات من كل صفحة
        enhanced_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # إنشاء مهام استخراج البيانات
            future_to_result = {executor.submit(self._scrape_page, result.url): result for result in base_response.results}
            
            for future in concurrent.futures.as_completed(future_to_result):
                result = future_to_result[future]
                
                try:
                    title, content, metadata = future.result()
                    
                    # تحديث النتيجة بالبيانات المستخرجة
                    if title:
                        result.title = title
                    
                    # تحديث المقتطف بجزء من المحتوى
                    if content:
                        # اختيار جزء من المحتوى كمقتطف
                        snippet_length = 200
                        if len(content) > snippet_length:
                            # البحث عن موضع ظهور استعلام البحث
                            query_pos = content.lower().find(config.query.lower())
                            if query_pos >= 0:
                                # اختيار المقتطف حول موضع ظهور الاستعلام
                                start = max(0, query_pos - snippet_length // 2)
                                end = min(len(content), start + snippet_length)
                                result.snippet = content[start:end] + "..."
                            else:
                                # اختيار بداية المحتوى كمقتطف
                                result.snippet = content[:snippet_length] + "..."
                        else:
                            result.snippet = content
                    
                    # تحديث البيانات الوصفية
                    result.metadata.update(metadata)
                    
                    # تحديث المؤلف والتاريخ إذا كانا متوفرين
                    if "author" in metadata:
                        result.author = metadata["author"]
                    
                    if "date" in metadata:
                        result.date = metadata["date"]
                    
                    enhanced_results.append(result)
                
                except Exception as e:
                    self.logger.error(f"فشل في معالجة النتيجة: {e}")
                    enhanced_results.append(result)
        
        # حساب وقت البحث
        search_time = time.time() - start_time
        
        return SearchResponse(
            query=config.query,
            results=enhanced_results,
            total_results=base_response.total_results,
            search_time=search_time,
            mode=config.mode,
            related_queries=base_response.related_queries,
            related_concepts=base_response.related_concepts,
            metadata={
                "engine": "WebScrapingEngine",
                "base_engine": base_response.metadata.get("engine", "Unknown"),
                "scraping_enabled": True
            }
        )
    
    def is_available(self) -> bool:
        """
        التحقق من توفر محرك البحث.
        
        Returns:
            True إذا كان محرك البحث متوفراً، وإلا False
        """
        try:
            import requests
            import bs4
            return True
        except ImportError:
            return False


class SemanticSearchEngine:
    """محرك بحث دلالي يدمج المتجهات الدلالية في عملية البحث."""
    
    def __init__(self):
        """تهيئة محرك البحث الدلالي."""
        self.logger = logging.getLogger('basira_intelligent_search.semantic')
        
        # تهيئة محركات البحث الأساسية
        self.engines = {
            "mock": MockSearchEngine(),
            "google": GoogleSearchEngine(),
            "web_scraping": WebScrapingEngine()
        }
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
        self.knowledge_generator = KnowledgeGenerator()
        self.conceptual_graph = ConceptualGraph()
    
    def _select_engine(self, config: SearchConfig) -> SearchEngineBase:
        """
        اختيار محرك البحث المناسب.
        
        Args:
            config: تكوين البحث
            
        Returns:
            محرك البحث المناسب
        """
        # التحقق من توفر محركات البحث
        available_engines = {name: engine for name, engine in self.engines.items() if engine.is_available()}
        
        if not available_engines:
            self.logger.warning("لا توجد محركات بحث متوفرة، استخدام محرك البحث الوهمي")
            return MockSearchEngine()
        
        # اختيار محرك البحث حسب النمط
        if config.mode == SearchMode.GENERAL:
            # تفضيل Google للبحث العام
            if "google" in available_engines:
                return available_engines["google"]
        
        elif config.mode == SearchMode.DEEP:
            # تفضيل استخراج البيانات من الويب للبحث العميق
            if "web_scraping" in available_engines:
                return available_engines["web_scraping"]
        
        # استخدام أي محرك بحث متوفر
        for name in ["google", "web_scraping", "mock"]:
            if name in available_engines:
                return available_engines[name]
        
        # استخدام محرك البحث الوهمي كملاذ أخير
        return MockSearchEngine()
    
    def _enhance_query_with_semantics(self, query: str, semantic_vector: Optional[SemanticVector] = None) -> str:
        """
        تحسين استعلام البحث باستخدام المعلومات الدلالية.
        
        Args:
            query: استعلام البحث الأصلي
            semantic_vector: المتجه الدلالي
            
        Returns:
            استعلام البحث المحسن
        """
        if semantic_vector is None:
            return query
        
        # استخراج الأبعاد الدلالية الرئيسية
        semantic_aspects = []
        for dim, value in semantic_vector.dimensions.items():
            if abs(value) > 0.5:  # اختيار الأبعاد ذات القيم العالية فقط
                aspect = dim.name.lower().replace('_', ' ')
                if value > 0:
                    semantic_aspects.append(aspect)
        
        # دمج الجوانب الدلالية مع الاستعلام
        if semantic_aspects:
            enhanced_query = f"{query} {' '.join(semantic_aspects)}"
            self.logger.info(f"تم تحسين الاستعلام: {enhanced_query}")
            return enhanced_query
        
        return query
    
    def _extract_concepts_from_results(self, response: SearchResponse) -> List[str]:
        """
        استخراج المفاهيم من نتائج البحث.
        
        Args:
            response: استجابة البحث
            
        Returns:
            قائمة بالمفاهيم المستخرجة
        """
        concepts = set()
        
        # استخراج المفاهيم من كل نتيجة
        for result in response.results:
            # استخراج المفاهيم من العنوان والمقتطف
            text = f"{result.title} {result.snippet}"
            
            # استخدام مستخرج المعرفة لاستخراج المفاهيم
            extracted_concepts = self.knowledge_extractor.extract_concepts(text)
            
            # إضافة المفاهيم المستخرجة
            concepts.update(extracted_concepts)
        
        return list(concepts)
    
    def _rank_results_semantically(self, results: List[SearchResult], semantic_vector: Optional[SemanticVector] = None) -> List[SearchResult]:
        """
        ترتيب النتائج دلالياً.
        
        Args:
            results: نتائج البحث
            semantic_vector: المتجه الدلالي
            
        Returns:
            نتائج البحث المرتبة دلالياً
        """
        if semantic_vector is None:
            return results
        
        # حساب المتجهات الدلالية للنتائج
        for result in results:
            if result.semantic_vector is None:
                # استخراج المتجه الدلالي من النص
                text = f"{result.title} {result.snippet}"
                result.semantic_vector = self.knowledge_extractor.extract_semantic_vector(text)
        
        # حساب درجات التشابه الدلالي
        for result in results:
            if result.semantic_vector is not None:
                # حساب تشابه جيب التمام بين المتجهين
                similarity = result.semantic_vector.cosine_similarity(semantic_vector)
                
                # تحديث درجة الصلة
                result.relevance_score = similarity
        
        # ترتيب النتائج حسب درجة الصلة
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def search(self, config: SearchConfig) -> SearchResponse:
        """
        إجراء بحث دلالي.
        
        Args:
            config: تكوين البحث
            
        Returns:
            استجابة البحث
        """
        # تحسين استعلام البحث باستخدام المعلومات الدلالية
        if config.semantic_vector is not None:
            config.query = self._enhance_query_with_semantics(config.query, config.semantic_vector)
        
        # اختيار محرك البحث المناسب
        engine = self._select_engine(config)
        self.logger.info(f"استخدام محرك بحث: {engine.__class__.__name__}")
        
        # إجراء البحث
        response = engine.search(config)
        
        # استخراج المفاهيم من النتائج
        if config.include_related_concepts:
            response.related_concepts = self._extract_concepts_from_results(response)
        
        # ترتيب النتائج دلالياً
        if config.semantic_vector is not None:
            response.results = self._rank_results_semantically(response.results, config.semantic_vector)
        
        return response
    
    def update_knowledge_graph(self, response: SearchResponse) -> None:
        """
        تحديث الرسم البياني المعرفي بناءً على نتائج البحث.
        
        Args:
            response: استجابة البحث
        """
        # استخراج المفاهيم من النتائج
        concepts = self._extract_concepts_from_results(response)
        
        # إضافة المفاهيم إلى الرسم البياني
        for concept_name in concepts:
            # التحقق من وجود المفهوم
            concept_exists = False
            for node_id, node in self.conceptual_graph.nodes.items():
                if node.name.lower() == concept_name.lower():
                    concept_exists = True
                    break
            
            # إضافة المفهوم إذا لم يكن موجوداً
            if not concept_exists:
                # إنشاء معرف فريد للمفهوم
                concept_id = f"concept_{len(self.conceptual_graph.nodes) + 1}"
                
                # استخراج المتجه الدلالي للمفهوم
                semantic_vector = self.knowledge_extractor.extract_semantic_vector(concept_name)
                
                # إنشاء عقدة المفهوم
                node = ConceptNode(
                    id=concept_id,
                    name=concept_name,
                    description=f"مفهوم مستخرج من البحث عن: {response.query}",
                    semantic_vector=semantic_vector
                )
                
                # إضافة المفهوم إلى الرسم البياني
                self.conceptual_graph.add_node(node)
        
        # إضافة العلاقات بين المفاهيم
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i != j:
                    # البحث عن معرفات المفاهيم
                    concept1_id = None
                    concept2_id = None
                    
                    for node_id, node in self.conceptual_graph.nodes.items():
                        if node.name.lower() == concept1.lower():
                            concept1_id = node_id
                        elif node.name.lower() == concept2.lower():
                            concept2_id = node_id
                    
                    # إضافة العلاقة إذا تم العثور على المفهومين
                    if concept1_id and concept2_id:
                        # التحقق من وجود العلاقة
                        relation_exists = False
                        for relation in self.conceptual_graph.nodes[concept1_id].relations.get("related_to", []):
                            if relation.target_id == concept2_id:
                                relation_exists = True
                                break
                        
                        # إضافة العلاقة إذا لم تكن موجودة
                        if not relation_exists:
                            relation = ConceptRelation(
                                source_id=concept1_id,
                                target_id=concept2_id,
                                relation_type="related_to",
                                weight=0.5  # وزن افتراضي
                            )
                            
                            self.conceptual_graph.add_relation(relation)


# تنفيذ الاختبار إذا تم تشغيل الملف مباشرة
if __name__ == "__main__":
    # إنشاء محرك البحث الدلالي
    engine = SemanticSearchEngine()
    
    # إنشاء تكوين البحث
    config = SearchConfig(
        query="الذكاء الاصطناعي في اللغة العربية",
        mode=SearchMode.GENERAL,
        content_types=[ContentType.TEXT],
        max_results=5
    )
    
    # إجراء البحث
    response = engine.search(config)
    
    # عرض النتائج
    print(f"نتائج البحث عن: {response.query}")
    print(f"عدد النتائج: {response.total_results}")
    print(f"وقت البحث: {response.search_time:.2f} ثانية")
    print("\nالنتائج:")
    
    for i, result in enumerate(response.results):
        print(f"\n{i+1}. {result.title}")
        print(f"   الرابط: {result.url}")
        print(f"   المقتطف: {result.snippet}")
        print(f"   درجة الصلة: {result.relevance_score:.2f}")
    
    print("\nالاستعلامات ذات الصلة:")
    for query in response.related_queries:
        print(f"- {query}")
    
    print("\nالمفاهيم ذات الصلة:")
    for concept in response.related_concepts:
        print(f"- {concept}")
    
    # تحديث الرسم البياني المعرفي
    engine.update_knowledge_graph(response)
    print(f"\nتم تحديث الرسم البياني المعرفي بـ {len(response.related_concepts)} مفهوم")
