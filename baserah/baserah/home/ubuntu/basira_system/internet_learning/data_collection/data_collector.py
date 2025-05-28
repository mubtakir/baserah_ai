#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة جمع البيانات من الإنترنت لنظام بصيرة

هذا الملف يحدد وحدة جمع البيانات من الإنترنت لنظام بصيرة، التي تمكن النظام من
جمع وتنظيم البيانات من مصادر متنوعة على الإنترنت لتغذية النظام المعرفي.

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
import csv
import pandas as pd
import sqlite3
from datetime import datetime

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import SemanticVector, ConceptualGraph, ConceptNode
from knowledge_extraction_generation import KnowledgeExtractor
from search.intelligent_search import SearchConfig, SearchMode, ContentType, SemanticSearchEngine, SearchResponse

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_data_collection')


class DataSourceType(Enum):
    """أنواع مصادر البيانات."""
    WEB_PAGE = auto()  # صفحة ويب
    API = auto()  # واجهة برمجة تطبيقات
    RSS_FEED = auto()  # تغذية RSS
    SOCIAL_MEDIA = auto()  # وسائل التواصل الاجتماعي
    DATASET = auto()  # مجموعة بيانات
    ACADEMIC = auto()  # مصدر أكاديمي
    NEWS = auto()  # مصدر إخباري
    FORUM = auto()  # منتدى
    BLOG = auto()  # مدونة
    CUSTOM = auto()  # مصدر مخصص


class DataFormat(Enum):
    """تنسيقات البيانات."""
    TEXT = auto()  # نص
    HTML = auto()  # HTML
    JSON = auto()  # JSON
    XML = auto()  # XML
    CSV = auto()  # CSV
    EXCEL = auto()  # Excel
    PDF = auto()  # PDF
    IMAGE = auto()  # صورة
    AUDIO = auto()  # صوت
    VIDEO = auto()  # فيديو
    BINARY = auto()  # ثنائي
    MIXED = auto()  # مختلط


@dataclass
class DataCollectionConfig:
    """تكوين جمع البيانات."""
    sources: List[str]  # مصادر البيانات (روابط، مسارات، إلخ)
    source_type: DataSourceType = DataSourceType.WEB_PAGE  # نوع المصدر
    expected_format: DataFormat = DataFormat.HTML  # التنسيق المتوقع
    max_items: int = 100  # أقصى عدد للعناصر
    depth: int = 1  # عمق الزحف (للصفحات المترابطة)
    follow_links: bool = False  # متابعة الروابط
    rate_limit: float = 1.0  # حد معدل الطلبات (بالثواني)
    timeout: int = 30  # مهلة الطلب (بالثواني)
    headers: Dict[str, str] = field(default_factory=dict)  # رؤوس HTTP
    params: Dict[str, Any] = field(default_factory=dict)  # معلمات الطلب
    auth: Optional[Tuple[str, str]] = None  # معلومات المصادقة (اسم المستخدم، كلمة المرور)
    proxy: Optional[Dict[str, str]] = None  # وكيل HTTP
    filters: Dict[str, Any] = field(default_factory=dict)  # مرشحات البيانات
    transformations: List[str] = field(default_factory=list)  # تحويلات البيانات
    storage_path: Optional[str] = None  # مسار التخزين
    additional_params: Dict[str, Any] = field(default_factory=dict)  # معلمات إضافية


@dataclass
class DataItem:
    """عنصر بيانات فردي."""
    content: Any  # محتوى البيانات
    source: str  # مصدر البيانات
    format: DataFormat  # تنسيق البيانات
    timestamp: float  # طابع زمني
    metadata: Dict[str, Any] = field(default_factory=dict)  # بيانات وصفية
    
    def to_dict(self) -> Dict[str, Any]:
        """
        تحويل عنصر البيانات إلى قاموس.
        
        Returns:
            قاموس يمثل عنصر البيانات
        """
        result = {
            "source": self.source,
            "format": self.format.name,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
        
        # معالجة المحتوى حسب التنسيق
        if self.format == DataFormat.TEXT:
            result["content"] = self.content
        elif self.format == DataFormat.JSON:
            if isinstance(self.content, dict) or isinstance(self.content, list):
                result["content"] = self.content
            else:
                result["content"] = str(self.content)
        else:
            result["content"] = str(self.content)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataItem':
        """
        إنشاء عنصر بيانات من قاموس.
        
        Args:
            data: قاموس يمثل عنصر البيانات
            
        Returns:
            عنصر البيانات
        """
        return cls(
            content=data["content"],
            source=data["source"],
            format=DataFormat[data["format"]],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


@dataclass
class DataCollectionResult:
    """نتيجة جمع البيانات."""
    items: List[DataItem]  # عناصر البيانات
    config: DataCollectionConfig  # تكوين جمع البيانات
    total_items: int  # إجمالي عدد العناصر
    collection_time: float  # وقت الجمع بالثواني
    success_rate: float  # معدل النجاح
    errors: List[str] = field(default_factory=list)  # أخطاء الجمع
    metadata: Dict[str, Any] = field(default_factory=dict)  # بيانات وصفية إضافية
    
    def save(self, path: Optional[str] = None) -> str:
        """
        حفظ نتيجة جمع البيانات.
        
        Args:
            path: مسار الحفظ (اختياري)
            
        Returns:
            مسار الملف المحفوظ
        """
        # استخدام المسار المحدد في التكوين إذا لم يتم توفير مسار
        if path is None:
            path = self.config.storage_path
        
        # استخدام مسار افتراضي إذا لم يتم توفير مسار
        if path is None:
            path = f"data_collection_result_{int(time.time())}.json"
        
        # التأكد من وجود المجلد
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # حفظ النتيجة
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "items": [item.to_dict() for item in self.items],
                "total_items": self.total_items,
                "collection_time": self.collection_time,
                "success_rate": self.success_rate,
                "errors": self.errors,
                "metadata": self.metadata,
                "config": {
                    "sources": self.config.sources,
                    "source_type": self.config.source_type.name,
                    "expected_format": self.config.expected_format.name,
                    "max_items": self.config.max_items,
                    "depth": self.config.depth,
                    "follow_links": self.config.follow_links,
                    "filters": self.config.filters,
                    "transformations": self.config.transformations
                }
            }, f, ensure_ascii=False, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'DataCollectionResult':
        """
        تحميل نتيجة جمع البيانات.
        
        Args:
            path: مسار الملف
            
        Returns:
            نتيجة جمع البيانات
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # إنشاء تكوين جمع البيانات
        config = DataCollectionConfig(
            sources=data["config"]["sources"],
            source_type=DataSourceType[data["config"]["source_type"]],
            expected_format=DataFormat[data["config"]["expected_format"]],
            max_items=data["config"]["max_items"],
            depth=data["config"]["depth"],
            follow_links=data["config"]["follow_links"],
            filters=data["config"].get("filters", {}),
            transformations=data["config"].get("transformations", [])
        )
        
        # إنشاء عناصر البيانات
        items = [DataItem.from_dict(item) for item in data["items"]]
        
        return cls(
            items=items,
            config=config,
            total_items=data["total_items"],
            collection_time=data["collection_time"],
            success_rate=data["success_rate"],
            errors=data["errors"],
            metadata=data.get("metadata", {})
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        تحويل نتيجة جمع البيانات إلى إطار بيانات.
        
        Returns:
            إطار بيانات يمثل نتيجة جمع البيانات
        """
        # إنشاء قائمة بالقواميس
        data = []
        for item in self.items:
            item_dict = {
                "source": item.source,
                "format": item.format.name,
                "timestamp": item.timestamp
            }
            
            # إضافة المحتوى
            if isinstance(item.content, (str, int, float, bool)):
                item_dict["content"] = item.content
            else:
                item_dict["content"] = str(item.content)
            
            # إضافة البيانات الوصفية
            for key, value in item.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    item_dict[f"metadata_{key}"] = value
            
            data.append(item_dict)
        
        # إنشاء إطار البيانات
        return pd.DataFrame(data)
    
    def to_sqlite(self, db_path: str, table_name: str = "data_items") -> None:
        """
        حفظ نتيجة جمع البيانات في قاعدة بيانات SQLite.
        
        Args:
            db_path: مسار قاعدة البيانات
            table_name: اسم الجدول
        """
        # تحويل النتيجة إلى إطار بيانات
        df = self.to_dataframe()
        
        # إنشاء اتصال بقاعدة البيانات
        conn = sqlite3.connect(db_path)
        
        # حفظ إطار البيانات في قاعدة البيانات
        df.to_sql(table_name, conn, if_exists='append', index=False)
        
        # إغلاق الاتصال
        conn.close()


class DataCollectorBase(ABC):
    """الفئة الأساسية لجامع البيانات."""
    
    def __init__(self):
        """تهيئة جامع البيانات."""
        self.logger = logging.getLogger('basira_data_collection.base')
    
    @abstractmethod
    def collect(self, config: DataCollectionConfig) -> DataCollectionResult:
        """
        جمع البيانات.
        
        Args:
            config: تكوين جمع البيانات
            
        Returns:
            نتيجة جمع البيانات
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        التحقق من توفر جامع البيانات.
        
        Returns:
            True إذا كان جامع البيانات متوفراً، وإلا False
        """
        pass


class WebPageCollector(DataCollectorBase):
    """جامع بيانات صفحات الويب."""
    
    def __init__(self):
        """تهيئة جامع بيانات صفحات الويب."""
        super().__init__()
        self.logger = logging.getLogger('basira_data_collection.web_page')
        self.session = requests.Session()
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """
        استخراج الروابط من HTML.
        
        Args:
            html: محتوى HTML
            base_url: الرابط الأساسي
            
        Returns:
            قائمة بالروابط المستخرجة
        """
        links = []
        
        try:
            # تحليل HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # استخراج الروابط
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                # تجاهل الروابط الفارغة والروابط الداخلية
                if not href or href.startswith('#'):
                    continue
                
                # تحويل الروابط النسبية إلى روابط مطلقة
                absolute_url = urljoin(base_url, href)
                
                # تجاهل الروابط غير HTTP/HTTPS
                if not absolute_url.startswith(('http://', 'https://')):
                    continue
                
                links.append(absolute_url)
        
        except Exception as e:
            self.logger.error(f"فشل في استخراج الروابط: {e}")
        
        return links
    
    def _fetch_url(self, url: str, config: DataCollectionConfig) -> Optional[DataItem]:
        """
        جلب محتوى رابط.
        
        Args:
            url: الرابط
            config: تكوين جمع البيانات
            
        Returns:
            عنصر البيانات إذا نجح الجلب، وإلا None
        """
        try:
            # إضافة تأخير لتجنب الحظر
            time.sleep(config.rate_limit)
            
            # إعداد رؤوس HTTP
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            headers.update(config.headers)
            
            # إجراء طلب HTTP
            response = self.session.get(
                url,
                headers=headers,
                params=config.params,
                timeout=config.timeout,
                auth=config.auth,
                proxies=config.proxy
            )
            response.raise_for_status()
            
            # تحديد تنسيق المحتوى
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'text/html' in content_type:
                format = DataFormat.HTML
                content = response.text
            elif 'application/json' in content_type:
                format = DataFormat.JSON
                content = response.json()
            elif 'text/xml' in content_type or 'application/xml' in content_type:
                format = DataFormat.XML
                content = response.text
            elif 'text/csv' in content_type:
                format = DataFormat.CSV
                content = response.text
            elif 'application/pdf' in content_type:
                format = DataFormat.PDF
                content = response.content
            elif 'image/' in content_type:
                format = DataFormat.IMAGE
                content = response.content
            elif 'video/' in content_type:
                format = DataFormat.VIDEO
                content = response.content
            elif 'audio/' in content_type:
                format = DataFormat.AUDIO
                content = response.content
            else:
                format = DataFormat.TEXT
                content = response.text
            
            # إنشاء عنصر البيانات
            return DataItem(
                content=content,
                source=url,
                format=format,
                timestamp=time.time(),
                metadata={
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "encoding": response.encoding,
                    "headers": dict(response.headers)
                }
            )
        
        except Exception as e:
            self.logger.error(f"فشل في جلب {url}: {e}")
            return None
    
    def collect(self, config: DataCollectionConfig) -> DataCollectionResult:
        """
        جمع بيانات صفحات الويب.
        
        Args:
            config: تكوين جمع البيانات
            
        Returns:
            نتيجة جمع البيانات
        """
        self.logger.info(f"بدء جمع البيانات من {len(config.sources)} مصدر")
        
        # قياس وقت الجمع
        start_time = time.time()
        
        # تهيئة النتائج
        items = []
        errors = []
        visited_urls = set()
        urls_to_visit = list(config.sources)
        
        # جمع البيانات
        while urls_to_visit and len(items) < config.max_items:
            # الحصول على الرابط التالي
            url = urls_to_visit.pop(0)
            
            # تجاهل الروابط المزارة
            if url in visited_urls:
                continue
            
            # إضافة الرابط إلى الروابط المزارة
            visited_urls.add(url)
            
            # جلب محتوى الرابط
            item = self._fetch_url(url, config)
            
            # معالجة النتيجة
            if item:
                # تطبيق المرشحات
                if self._apply_filters(item, config.filters):
                    # تطبيق التحويلات
                    item = self._apply_transformations(item, config.transformations)
                    
                    # إضافة العنصر إلى النتائج
                    items.append(item)
                
                # استخراج الروابط إذا كان متابعة الروابط مفعلة وعمق الزحف أكبر من 0
                if config.follow_links and config.depth > 0 and item.format == DataFormat.HTML:
                    # استخراج الروابط
                    links = self._extract_links(item.content, url)
                    
                    # إضافة الروابط إلى قائمة الزيارة
                    for link in links:
                        if link not in visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)
                    
                    # تقليل عمق الزحف
                    config.depth -= 1
            else:
                # إضافة الخطأ
                errors.append(f"فشل في جلب {url}")
        
        # حساب وقت الجمع
        collection_time = time.time() - start_time
        
        # حساب معدل النجاح
        success_rate = len(items) / len(visited_urls) if visited_urls else 0.0
        
        return DataCollectionResult(
            items=items,
            config=config,
            total_items=len(items),
            collection_time=collection_time,
            success_rate=success_rate,
            errors=errors,
            metadata={
                "collector": "WebPageCollector",
                "visited_urls": len(visited_urls),
                "remaining_urls": len(urls_to_visit)
            }
        )
    
    def _apply_filters(self, item: DataItem, filters: Dict[str, Any]) -> bool:
        """
        تطبيق المرشحات على عنصر البيانات.
        
        Args:
            item: عنصر البيانات
            filters: المرشحات
            
        Returns:
            True إذا اجتاز العنصر المرشحات، وإلا False
        """
        # إذا لم تكن هناك مرشحات، اجتاز العنصر
        if not filters:
            return True
        
        # مرشح التنسيق
        if "format" in filters and item.format.name != filters["format"]:
            return False
        
        # مرشح المصدر
        if "source" in filters and not item.source.startswith(filters["source"]):
            return False
        
        # مرشح المحتوى
        if "content_contains" in filters:
            if item.format == DataFormat.TEXT or item.format == DataFormat.HTML:
                if filters["content_contains"] not in item.content:
                    return False
        
        # مرشح الحجم
        if "min_size" in filters:
            if item.format == DataFormat.TEXT or item.format == DataFormat.HTML:
                if len(item.content) < filters["min_size"]:
                    return False
        
        if "max_size" in filters:
            if item.format == DataFormat.TEXT or item.format == DataFormat.HTML:
                if len(item.content) > filters["max_size"]:
                    return False
        
        return True
    
    def _apply_transformations(self, item: DataItem, transformations: List[str]) -> DataItem:
        """
        تطبيق التحويلات على عنصر البيانات.
        
        Args:
            item: عنصر البيانات
            transformations: التحويلات
            
        Returns:
            عنصر البيانات بعد التحويلات
        """
        # إذا لم تكن هناك تحويلات، إرجاع العنصر كما هو
        if not transformations:
            return item
        
        # نسخة من العنصر
        transformed_item = item
        
        # تطبيق التحويلات
        for transformation in transformations:
            if transformation == "html_to_text" and transformed_item.format == DataFormat.HTML:
                # تحويل HTML إلى نص
                soup = BeautifulSoup(transformed_item.content, 'html.parser')
                transformed_item.content = soup.get_text()
                transformed_item.format = DataFormat.TEXT
            
            elif transformation == "extract_title" and transformed_item.format == DataFormat.HTML:
                # استخراج العنوان
                soup = BeautifulSoup(transformed_item.content, 'html.parser')
                title = soup.title.string if soup.title else ""
                transformed_item.metadata["title"] = title
            
            elif transformation == "extract_metadata" and transformed_item.format == DataFormat.HTML:
                # استخراج البيانات الوصفية
                soup = BeautifulSoup(transformed_item.content, 'html.parser')
                
                # استخراج الكلمات المفتاحية
                meta_keywords = soup.find("meta", attrs={"name": "keywords"})
                if meta_keywords:
                    transformed_item.metadata["keywords"] = meta_keywords.get("content", "").split(",")
                
                # استخراج الوصف
                meta_description = soup.find("meta", attrs={"name": "description"})
                if meta_description:
                    transformed_item.metadata["description"] = meta_description.get("content", "")
                
                # استخراج المؤلف
                meta_author = soup.find("meta", attrs={"name": "author"})
                if meta_author:
                    transformed_item.metadata["author"] = meta_author.get("content", "")
                
                # استخراج التاريخ
                meta_date = soup.find("meta", attrs={"name": "date"})
                if meta_date:
                    transformed_item.metadata["date"] = meta_date.get("content", "")
        
        return transformed_item
    
    def is_available(self) -> bool:
        """
        التحقق من توفر جامع البيانات.
        
        Returns:
            True إذا كان جامع البيانات متوفراً، وإلا False
        """
        try:
            import requests
            import bs4
            return True
        except ImportError:
            return False


class APICollector(DataCollectorBase):
    """جامع بيانات واجهات برمجة التطبيقات."""
    
    def __init__(self):
        """تهيئة جامع بيانات واجهات برمجة التطبيقات."""
        super().__init__()
        self.logger = logging.getLogger('basira_data_collection.api')
        self.session = requests.Session()
    
    def collect(self, config: DataCollectionConfig) -> DataCollectionResult:
        """
        جمع بيانات واجهات برمجة التطبيقات.
        
        Args:
            config: تكوين جمع البيانات
            
        Returns:
            نتيجة جمع البيانات
        """
        self.logger.info(f"بدء جمع البيانات من {len(config.sources)} مصدر API")
        
        # قياس وقت الجمع
        start_time = time.time()
        
        # تهيئة النتائج
        items = []
        errors = []
        
        # جمع البيانات
        for url in config.sources:
            try:
                # إضافة تأخير لتجنب الحظر
                time.sleep(config.rate_limit)
                
                # إعداد رؤوس HTTP
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "application/json"
                }
                headers.update(config.headers)
                
                # إجراء طلب HTTP
                response = self.session.get(
                    url,
                    headers=headers,
                    params=config.params,
                    timeout=config.timeout,
                    auth=config.auth,
                    proxies=config.proxy
                )
                response.raise_for_status()
                
                # تحليل الاستجابة
                try:
                    content = response.json()
                    format = DataFormat.JSON
                except ValueError:
                    content = response.text
                    format = DataFormat.TEXT
                
                # إنشاء عنصر البيانات
                item = DataItem(
                    content=content,
                    source=url,
                    format=format,
                    timestamp=time.time(),
                    metadata={
                        "status_code": response.status_code,
                        "content_type": response.headers.get('Content-Type', ''),
                        "encoding": response.encoding,
                        "headers": dict(response.headers)
                    }
                )
                
                # تطبيق المرشحات
                if self._apply_filters(item, config.filters):
                    # تطبيق التحويلات
                    item = self._apply_transformations(item, config.transformations)
                    
                    # إضافة العنصر إلى النتائج
                    items.append(item)
                    
                    # التحقق من الوصول إلى الحد الأقصى
                    if len(items) >= config.max_items:
                        break
            
            except Exception as e:
                self.logger.error(f"فشل في جلب {url}: {e}")
                errors.append(f"فشل في جلب {url}: {e}")
        
        # حساب وقت الجمع
        collection_time = time.time() - start_time
        
        # حساب معدل النجاح
        success_rate = len(items) / len(config.sources) if config.sources else 0.0
        
        return DataCollectionResult(
            items=items,
            config=config,
            total_items=len(items),
            collection_time=collection_time,
            success_rate=success_rate,
            errors=errors,
            metadata={
                "collector": "APICollector"
            }
        )
    
    def _apply_filters(self, item: DataItem, filters: Dict[str, Any]) -> bool:
        """
        تطبيق المرشحات على عنصر البيانات.
        
        Args:
            item: عنصر البيانات
            filters: المرشحات
            
        Returns:
            True إذا اجتاز العنصر المرشحات، وإلا False
        """
        # إذا لم تكن هناك مرشحات، اجتاز العنصر
        if not filters:
            return True
        
        # مرشح التنسيق
        if "format" in filters and item.format.name != filters["format"]:
            return False
        
        # مرشح المصدر
        if "source" in filters and not item.source.startswith(filters["source"]):
            return False
        
        # مرشح المحتوى JSON
        if "json_path" in filters and item.format == DataFormat.JSON:
            try:
                # تقسيم المسار
                path_parts = filters["json_path"].split('.')
                
                # الوصول إلى العنصر
                value = item.content
                for part in path_parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    elif isinstance(value, list) and part.isdigit():
                        index = int(part)
                        if 0 <= index < len(value):
                            value = value[index]
                        else:
                            return False
                    else:
                        return False
                
                # التحقق من القيمة
                if "json_value" in filters and value != filters["json_value"]:
                    return False
            except Exception:
                return False
        
        return True
    
    def _apply_transformations(self, item: DataItem, transformations: List[str]) -> DataItem:
        """
        تطبيق التحويلات على عنصر البيانات.
        
        Args:
            item: عنصر البيانات
            transformations: التحويلات
            
        Returns:
            عنصر البيانات بعد التحويلات
        """
        # إذا لم تكن هناك تحويلات، إرجاع العنصر كما هو
        if not transformations:
            return item
        
        # نسخة من العنصر
        transformed_item = item
        
        # تطبيق التحويلات
        for transformation in transformations:
            if transformation == "flatten_json" and transformed_item.format == DataFormat.JSON:
                # تسطيح JSON
                flattened = {}
                
                def flatten(obj, name=''):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            flatten(value, f"{name}.{key}" if name else key)
                    elif isinstance(obj, list):
                        for i, value in enumerate(obj):
                            flatten(value, f"{name}[{i}]")
                    else:
                        flattened[name] = obj
                
                flatten(transformed_item.content)
                transformed_item.content = flattened
            
            elif transformation == "extract_json_fields" and transformed_item.format == DataFormat.JSON:
                # استخراج حقول محددة من JSON
                if "json_fields" in transformed_item.metadata:
                    fields = transformed_item.metadata["json_fields"]
                    extracted = {}
                    
                    for field in fields:
                        # تقسيم المسار
                        path_parts = field.split('.')
                        
                        # الوصول إلى العنصر
                        value = transformed_item.content
                        for part in path_parts:
                            if isinstance(value, dict):
                                value = value.get(part)
                            elif isinstance(value, list) and part.isdigit():
                                index = int(part)
                                if 0 <= index < len(value):
                                    value = value[index]
                                else:
                                    value = None
                                    break
                            else:
                                value = None
                                break
                        
                        # إضافة القيمة
                        extracted[field] = value
                    
                    transformed_item.content = extracted
        
        return transformed_item
    
    def is_available(self) -> bool:
        """
        التحقق من توفر جامع البيانات.
        
        Returns:
            True إذا كان جامع البيانات متوفراً، وإلا False
        """
        try:
            import requests
            return True
        except ImportError:
            return False


class RSSFeedCollector(DataCollectorBase):
    """جامع بيانات تغذيات RSS."""
    
    def __init__(self):
        """تهيئة جامع بيانات تغذيات RSS."""
        super().__init__()
        self.logger = logging.getLogger('basira_data_collection.rss_feed')
    
    def collect(self, config: DataCollectionConfig) -> DataCollectionResult:
        """
        جمع بيانات تغذيات RSS.
        
        Args:
            config: تكوين جمع البيانات
            
        Returns:
            نتيجة جمع البيانات
        """
        self.logger.info(f"بدء جمع البيانات من {len(config.sources)} تغذية RSS")
        
        # قياس وقت الجمع
        start_time = time.time()
        
        # تهيئة النتائج
        items = []
        errors = []
        
        try:
            import feedparser
            
            # جمع البيانات
            for url in config.sources:
                try:
                    # إضافة تأخير لتجنب الحظر
                    time.sleep(config.rate_limit)
                    
                    # تحليل التغذية
                    feed = feedparser.parse(url)
                    
                    # التحقق من وجود أخطاء
                    if hasattr(feed, 'bozo_exception'):
                        self.logger.warning(f"تحذير عند تحليل {url}: {feed.bozo_exception}")
                    
                    # معالجة العناصر
                    for entry in feed.entries[:config.max_items]:
                        # إنشاء عنصر البيانات
                        item = DataItem(
                            content=entry,
                            source=url,
                            format=DataFormat.JSON,
                            timestamp=time.time(),
                            metadata={
                                "feed_title": feed.feed.get('title', ''),
                                "feed_link": feed.feed.get('link', ''),
                                "feed_description": feed.feed.get('description', ''),
                                "entry_title": entry.get('title', ''),
                                "entry_link": entry.get('link', ''),
                                "entry_published": entry.get('published', '')
                            }
                        )
                        
                        # تطبيق المرشحات
                        if self._apply_filters(item, config.filters):
                            # تطبيق التحويلات
                            item = self._apply_transformations(item, config.transformations)
                            
                            # إضافة العنصر إلى النتائج
                            items.append(item)
                            
                            # التحقق من الوصول إلى الحد الأقصى
                            if len(items) >= config.max_items:
                                break
                
                except Exception as e:
                    self.logger.error(f"فشل في جلب {url}: {e}")
                    errors.append(f"فشل في جلب {url}: {e}")
        
        except ImportError:
            self.logger.error("مكتبة feedparser غير متوفرة")
            errors.append("مكتبة feedparser غير متوفرة")
        
        # حساب وقت الجمع
        collection_time = time.time() - start_time
        
        # حساب معدل النجاح
        success_rate = len(items) / len(config.sources) if config.sources else 0.0
        
        return DataCollectionResult(
            items=items,
            config=config,
            total_items=len(items),
            collection_time=collection_time,
            success_rate=success_rate,
            errors=errors,
            metadata={
                "collector": "RSSFeedCollector"
            }
        )
    
    def _apply_filters(self, item: DataItem, filters: Dict[str, Any]) -> bool:
        """
        تطبيق المرشحات على عنصر البيانات.
        
        Args:
            item: عنصر البيانات
            filters: المرشحات
            
        Returns:
            True إذا اجتاز العنصر المرشحات، وإلا False
        """
        # إذا لم تكن هناك مرشحات، اجتاز العنصر
        if not filters:
            return True
        
        # مرشح المصدر
        if "source" in filters and not item.source.startswith(filters["source"]):
            return False
        
        # مرشح العنوان
        if "title_contains" in filters:
            title = item.metadata.get("entry_title", "")
            if filters["title_contains"] not in title:
                return False
        
        # مرشح الوصف
        if "description_contains" in filters:
            description = item.metadata.get("feed_description", "")
            if filters["description_contains"] not in description:
                return False
        
        # مرشح التاريخ
        if "published_after" in filters:
            published = item.metadata.get("entry_published", "")
            if published:
                try:
                    published_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
                    after_date = datetime.strptime(filters["published_after"], "%Y-%m-%d")
                    if published_date.replace(tzinfo=None) < after_date:
                        return False
                except ValueError:
                    pass
        
        return True
    
    def _apply_transformations(self, item: DataItem, transformations: List[str]) -> DataItem:
        """
        تطبيق التحويلات على عنصر البيانات.
        
        Args:
            item: عنصر البيانات
            transformations: التحويلات
            
        Returns:
            عنصر البيانات بعد التحويلات
        """
        # إذا لم تكن هناك تحويلات، إرجاع العنصر كما هو
        if not transformations:
            return item
        
        # نسخة من العنصر
        transformed_item = item
        
        # تطبيق التحويلات
        for transformation in transformations:
            if transformation == "extract_content" and transformed_item.format == DataFormat.JSON:
                # استخراج المحتوى
                entry = transformed_item.content
                
                # استخراج المحتوى
                content = ""
                if 'content' in entry:
                    for content_item in entry.content:
                        content += content_item.value
                elif 'summary' in entry:
                    content = entry.summary
                elif 'description' in entry:
                    content = entry.description
                
                # تحويل HTML إلى نص
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text()
                
                # تحديث المحتوى
                transformed_item.content = content
                transformed_item.format = DataFormat.TEXT
            
            elif transformation == "extract_metadata" and transformed_item.format == DataFormat.JSON:
                # استخراج البيانات الوصفية
                entry = transformed_item.content
                
                # استخراج البيانات الوصفية
                metadata = {}
                
                # العنوان
                if 'title' in entry:
                    metadata["title"] = entry.title
                
                # الرابط
                if 'link' in entry:
                    metadata["link"] = entry.link
                
                # تاريخ النشر
                if 'published' in entry:
                    metadata["published"] = entry.published
                
                # المؤلف
                if 'author' in entry:
                    metadata["author"] = entry.author
                
                # الوسوم
                if 'tags' in entry:
                    metadata["tags"] = [tag.term for tag in entry.tags]
                
                # تحديث البيانات الوصفية
                transformed_item.metadata.update(metadata)
        
        return transformed_item
    
    def is_available(self) -> bool:
        """
        التحقق من توفر جامع البيانات.
        
        Returns:
            True إذا كان جامع البيانات متوفراً، وإلا False
        """
        try:
            import feedparser
            return True
        except ImportError:
            return False


class SemanticDataCollector:
    """جامع بيانات دلالي يدمج المتجهات الدلالية في عملية جمع البيانات."""
    
    def __init__(self):
        """تهيئة جامع البيانات الدلالي."""
        self.logger = logging.getLogger('basira_data_collection.semantic')
        
        # تهيئة جامعي البيانات الأساسيين
        self.collectors = {
            "web_page": WebPageCollector(),
            "api": APICollector(),
            "rss_feed": RSSFeedCollector()
        }
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
        self.search_engine = SemanticSearchEngine()
        self.conceptual_graph = ConceptualGraph()
    
    def _select_collector(self, config: DataCollectionConfig) -> DataCollectorBase:
        """
        اختيار جامع البيانات المناسب.
        
        Args:
            config: تكوين جمع البيانات
            
        Returns:
            جامع البيانات المناسب
        """
        # التحقق من توفر جامعي البيانات
        available_collectors = {name: collector for name, collector in self.collectors.items() if collector.is_available()}
        
        if not available_collectors:
            self.logger.warning("لا توجد جامعي بيانات متوفرة")
            return WebPageCollector()
        
        # اختيار جامع البيانات حسب نوع المصدر
        if config.source_type == DataSourceType.WEB_PAGE:
            # تفضيل جامع صفحات الويب
            if "web_page" in available_collectors:
                return available_collectors["web_page"]
        
        elif config.source_type == DataSourceType.API:
            # تفضيل جامع واجهات برمجة التطبيقات
            if "api" in available_collectors:
                return available_collectors["api"]
        
        elif config.source_type == DataSourceType.RSS_FEED:
            # تفضيل جامع تغذيات RSS
            if "rss_feed" in available_collectors:
                return available_collectors["rss_feed"]
        
        # استخدام أي جامع بيانات متوفر
        for name in ["web_page", "api", "rss_feed"]:
            if name in available_collectors:
                return available_collectors[name]
        
        # استخدام جامع صفحات الويب كملاذ أخير
        return WebPageCollector()
    
    def _discover_sources(self, query: str, max_sources: int = 10) -> List[str]:
        """
        اكتشاف مصادر البيانات باستخدام محرك البحث.
        
        Args:
            query: استعلام البحث
            max_sources: أقصى عدد للمصادر
            
        Returns:
            قائمة بمصادر البيانات
        """
        # إنشاء تكوين البحث
        search_config = SearchConfig(
            query=query,
            mode=SearchMode.GENERAL,
            content_types=[ContentType.TEXT],
            max_results=max_sources
        )
        
        # إجراء البحث
        response = self.search_engine.search(search_config)
        
        # استخراج الروابط
        sources = [result.url for result in response.results]
        
        return sources
    
    def _extract_semantic_vectors(self, result: DataCollectionResult) -> DataCollectionResult:
        """
        استخراج المتجهات الدلالية من نتيجة جمع البيانات.
        
        Args:
            result: نتيجة جمع البيانات
            
        Returns:
            نتيجة جمع البيانات مع المتجهات الدلالية
        """
        # استخراج المتجهات الدلالية لكل عنصر
        for item in result.items:
            # استخراج النص
            text = ""
            
            if item.format == DataFormat.TEXT:
                text = item.content
            elif item.format == DataFormat.HTML:
                # تحويل HTML إلى نص
                soup = BeautifulSoup(item.content, 'html.parser')
                text = soup.get_text()
            elif item.format == DataFormat.JSON:
                # تحويل JSON إلى نص
                if isinstance(item.content, dict):
                    text = json.dumps(item.content, ensure_ascii=False)
                else:
                    text = str(item.content)
            
            # استخراج المتجه الدلالي
            if text:
                semantic_vector = self.knowledge_extractor.extract_semantic_vector(text)
                item.metadata["semantic_vector"] = semantic_vector.to_dict()
        
        return result
    
    def _update_knowledge_graph(self, result: DataCollectionResult) -> None:
        """
        تحديث الرسم البياني المعرفي بناءً على نتيجة جمع البيانات.
        
        Args:
            result: نتيجة جمع البيانات
        """
        # استخراج المفاهيم من كل عنصر
        for item in result.items:
            # استخراج النص
            text = ""
            
            if item.format == DataFormat.TEXT:
                text = item.content
            elif item.format == DataFormat.HTML:
                # تحويل HTML إلى نص
                soup = BeautifulSoup(item.content, 'html.parser')
                text = soup.get_text()
            elif item.format == DataFormat.JSON:
                # تحويل JSON إلى نص
                if isinstance(item.content, dict):
                    text = json.dumps(item.content, ensure_ascii=False)
                else:
                    text = str(item.content)
            
            # استخراج المفاهيم
            if text:
                concepts = self.knowledge_extractor.extract_concepts(text)
                
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
                            description=f"مفهوم مستخرج من {item.source}",
                            semantic_vector=semantic_vector
                        )
                        
                        # إضافة المفهوم إلى الرسم البياني
                        self.conceptual_graph.add_node(node)
    
    def collect(self, config: DataCollectionConfig) -> DataCollectionResult:
        """
        جمع البيانات باستخدام المعلومات الدلالية.
        
        Args:
            config: تكوين جمع البيانات
            
        Returns:
            نتيجة جمع البيانات
        """
        # اكتشاف مصادر البيانات إذا لم يتم توفيرها
        if not config.sources and "query" in config.additional_params:
            config.sources = self._discover_sources(config.additional_params["query"], config.max_items)
        
        # اختيار جامع البيانات المناسب
        collector = self._select_collector(config)
        self.logger.info(f"استخدام جامع بيانات: {collector.__class__.__name__}")
        
        # جمع البيانات
        result = collector.collect(config)
        
        # استخراج المتجهات الدلالية
        result = self._extract_semantic_vectors(result)
        
        # تحديث الرسم البياني المعرفي
        self._update_knowledge_graph(result)
        
        return result


# تنفيذ الاختبار إذا تم تشغيل الملف مباشرة
if __name__ == "__main__":
    # إنشاء جامع البيانات الدلالي
    collector = SemanticDataCollector()
    
    # إنشاء تكوين جمع البيانات
    config = DataCollectionConfig(
        sources=["https://ar.wikipedia.org/wiki/ذكاء_اصطناعي"],
        source_type=DataSourceType.WEB_PAGE,
        expected_format=DataFormat.HTML,
        max_items=5,
        depth=1,
        follow_links=True,
        transformations=["html_to_text", "extract_metadata"],
        storage_path="data_collection_result.json"
    )
    
    # جمع البيانات
    result = collector.collect(config)
    
    # عرض النتائج
    print(f"تم جمع {result.total_items} عنصر بيانات")
    print(f"وقت الجمع: {result.collection_time:.2f} ثانية")
    print(f"معدل النجاح: {result.success_rate:.2f}")
    
    # حفظ النتائج
    saved_path = result.save()
    print(f"تم حفظ النتائج في: {saved_path}")
    
    # تحويل النتائج إلى إطار بيانات
    df = result.to_dataframe()
    print("\nإطار البيانات:")
    print(df.head())
