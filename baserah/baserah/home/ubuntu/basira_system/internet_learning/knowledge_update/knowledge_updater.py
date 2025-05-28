#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة تحديث المعرفة من الإنترنت لنظام بصيرة

هذا الملف يحدد وحدة تحديث المعرفة من الإنترنت لنظام بصيرة، التي تمكن النظام من
تحديث قاعدة معرفته بشكل مستمر من مصادر الإنترنت المختلفة.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import threading
import queue

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import SemanticVector, ConceptualGraph, ConceptNode, ConceptRelation
from knowledge_extraction_generation import KnowledgeExtractor, KnowledgeGenerator
from search.intelligent_search import SearchConfig, SearchMode, ContentType, SemanticSearchEngine, SearchResponse
from data_collection.data_collector import DataCollectionConfig, DataSourceType, DataFormat, SemanticDataCollector

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_knowledge_update')


class UpdatePriority(Enum):
    """أولويات تحديث المعرفة."""
    LOW = auto()  # منخفضة
    MEDIUM = auto()  # متوسطة
    HIGH = auto()  # عالية
    CRITICAL = auto()  # حرجة


class UpdateFrequency(Enum):
    """تواتر تحديث المعرفة."""
    HOURLY = auto()  # كل ساعة
    DAILY = auto()  # يومياً
    WEEKLY = auto()  # أسبوعياً
    MONTHLY = auto()  # شهرياً
    QUARTERLY = auto()  # ربع سنوي
    YEARLY = auto()  # سنوياً
    ONCE = auto()  # مرة واحدة
    ON_DEMAND = auto()  # عند الطلب


@dataclass
class KnowledgeUpdateConfig:
    """تكوين تحديث المعرفة."""
    topics: List[str]  # المواضيع المراد تحديثها
    frequency: UpdateFrequency = UpdateFrequency.DAILY  # تواتر التحديث
    priority: UpdatePriority = UpdatePriority.MEDIUM  # أولوية التحديث
    max_sources: int = 10  # أقصى عدد للمصادر
    max_concepts: int = 100  # أقصى عدد للمفاهيم
    update_depth: int = 2  # عمق التحديث
    search_modes: List[SearchMode] = field(default_factory=lambda: [SearchMode.GENERAL])  # أنماط البحث
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.TEXT])  # أنواع المحتوى
    source_types: List[DataSourceType] = field(default_factory=lambda: [DataSourceType.WEB_PAGE])  # أنواع المصادر
    update_existing: bool = True  # تحديث المفاهيم الموجودة
    add_new_concepts: bool = True  # إضافة مفاهيم جديدة
    verify_information: bool = True  # التحقق من المعلومات
    storage_path: Optional[str] = None  # مسار تخزين نتائج التحديث
    additional_params: Dict[str, Any] = field(default_factory=dict)  # معلمات إضافية


@dataclass
class KnowledgeUpdateTask:
    """مهمة تحديث المعرفة."""
    config: KnowledgeUpdateConfig  # تكوين التحديث
    task_id: str  # معرف المهمة
    created_at: datetime = field(default_factory=datetime.now)  # تاريخ الإنشاء
    scheduled_at: Optional[datetime] = None  # تاريخ الجدولة
    started_at: Optional[datetime] = None  # تاريخ البدء
    completed_at: Optional[datetime] = None  # تاريخ الإكمال
    status: str = "pending"  # حالة المهمة
    progress: float = 0.0  # تقدم المهمة
    result: Optional[Dict[str, Any]] = None  # نتيجة المهمة
    error: Optional[str] = None  # خطأ المهمة
    
    def to_dict(self) -> Dict[str, Any]:
        """
        تحويل المهمة إلى قاموس.
        
        Returns:
            قاموس يمثل المهمة
        """
        return {
            "task_id": self.task_id,
            "topics": self.config.topics,
            "frequency": self.config.frequency.name,
            "priority": self.config.priority.name,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error
        }


class KnowledgeUpdateManager:
    """مدير تحديث المعرفة."""
    
    def __init__(self, storage_dir: str = "knowledge_updates"):
        """
        تهيئة مدير تحديث المعرفة.
        
        Args:
            storage_dir: مجلد تخزين نتائج التحديث
        """
        self.logger = logging.getLogger('basira_knowledge_update.manager')
        self.storage_dir = storage_dir
        
        # إنشاء مجلد التخزين إذا لم يكن موجوداً
        os.makedirs(storage_dir, exist_ok=True)
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
        self.knowledge_generator = KnowledgeGenerator()
        self.search_engine = SemanticSearchEngine()
        self.data_collector = SemanticDataCollector()
        self.conceptual_graph = ConceptualGraph()
        
        # تهيئة قائمة المهام
        self.tasks: Dict[str, KnowledgeUpdateTask] = {}
        
        # تهيئة قائمة المهام المجدولة
        self.scheduled_tasks: List[KnowledgeUpdateTask] = []
        
        # تهيئة قائمة انتظار المهام
        self.task_queue = queue.Queue()
        
        # تهيئة مؤشر التشغيل
        self.running = False
        
        # تهيئة مؤشر الإيقاف
        self.stop_event = threading.Event()
        
        # تهيئة خيط العمل
        self.worker_thread = None
    
    def start(self) -> None:
        """بدء مدير تحديث المعرفة."""
        if self.running:
            self.logger.warning("مدير تحديث المعرفة قيد التشغيل بالفعل")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # بدء خيط العمل
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        self.logger.info("تم بدء مدير تحديث المعرفة")
    
    def stop(self) -> None:
        """إيقاف مدير تحديث المعرفة."""
        if not self.running:
            self.logger.warning("مدير تحديث المعرفة متوقف بالفعل")
            return
        
        self.running = False
        self.stop_event.set()
        
        # انتظار انتهاء خيط العمل
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        self.logger.info("تم إيقاف مدير تحديث المعرفة")
    
    def _worker_loop(self) -> None:
        """حلقة عمل مدير تحديث المعرفة."""
        while self.running and not self.stop_event.is_set():
            try:
                # التحقق من المهام المجدولة
                self._check_scheduled_tasks()
                
                # الحصول على المهمة التالية
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # تنفيذ المهمة
                self._execute_task(task)
                
                # تحديد المهمة كمكتملة
                self.task_queue.task_done()
            
            except Exception as e:
                self.logger.error(f"خطأ في حلقة العمل: {e}")
                time.sleep(5.0)
    
    def _check_scheduled_tasks(self) -> None:
        """التحقق من المهام المجدولة."""
        now = datetime.now()
        
        # التحقق من كل مهمة مجدولة
        for task in list(self.scheduled_tasks):
            if task.scheduled_at and task.scheduled_at <= now:
                # إضافة المهمة إلى قائمة الانتظار
                self.task_queue.put(task)
                
                # إزالة المهمة من قائمة المهام المجدولة
                self.scheduled_tasks.remove(task)
                
                # جدولة المهمة التالية إذا كانت دورية
                if task.config.frequency != UpdateFrequency.ONCE and task.config.frequency != UpdateFrequency.ON_DEMAND:
                    # إنشاء مهمة جديدة
                    next_task = KnowledgeUpdateTask(
                        config=task.config,
                        task_id=f"{task.task_id}_next",
                        created_at=now
                    )
                    
                    # تحديد موعد المهمة التالية
                    if task.config.frequency == UpdateFrequency.HOURLY:
                        next_task.scheduled_at = now + timedelta(hours=1)
                    elif task.config.frequency == UpdateFrequency.DAILY:
                        next_task.scheduled_at = now + timedelta(days=1)
                    elif task.config.frequency == UpdateFrequency.WEEKLY:
                        next_task.scheduled_at = now + timedelta(weeks=1)
                    elif task.config.frequency == UpdateFrequency.MONTHLY:
                        next_task.scheduled_at = now + timedelta(days=30)
                    elif task.config.frequency == UpdateFrequency.QUARTERLY:
                        next_task.scheduled_at = now + timedelta(days=90)
                    elif task.config.frequency == UpdateFrequency.YEARLY:
                        next_task.scheduled_at = now + timedelta(days=365)
                    
                    # إضافة المهمة إلى قائمة المهام المجدولة
                    self.scheduled_tasks.append(next_task)
                    self.tasks[next_task.task_id] = next_task
    
    def _execute_task(self, task: KnowledgeUpdateTask) -> None:
        """
        تنفيذ مهمة تحديث المعرفة.
        
        Args:
            task: مهمة التحديث
        """
        self.logger.info(f"بدء تنفيذ المهمة {task.task_id}")
        
        # تحديث حالة المهمة
        task.status = "running"
        task.started_at = datetime.now()
        task.progress = 0.0
        
        try:
            # تنفيذ التحديث
            result = self._update_knowledge(task.config)
            
            # تحديث حالة المهمة
            task.status = "completed"
            task.completed_at = datetime.now()
            task.progress = 1.0
            task.result = result
            
            self.logger.info(f"تم إكمال المهمة {task.task_id}")
        
        except Exception as e:
            # تحديث حالة المهمة
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error = str(e)
            
            self.logger.error(f"فشل في تنفيذ المهمة {task.task_id}: {e}")
    
    def _update_knowledge(self, config: KnowledgeUpdateConfig) -> Dict[str, Any]:
        """
        تحديث المعرفة.
        
        Args:
            config: تكوين التحديث
            
        Returns:
            نتيجة التحديث
        """
        results = {
            "topics": config.topics,
            "searches": [],
            "collections": [],
            "concepts_added": 0,
            "concepts_updated": 0,
            "relations_added": 0,
            "total_sources": 0
        }
        
        # البحث عن كل موضوع
        for topic in config.topics:
            # البحث باستخدام كل نمط بحث
            for search_mode in config.search_modes:
                # إنشاء تكوين البحث
                search_config = SearchConfig(
                    query=topic,
                    mode=search_mode,
                    content_types=config.content_types,
                    max_results=config.max_sources
                )
                
                # إجراء البحث
                search_response = self.search_engine.search(search_config)
                
                # إضافة نتيجة البحث إلى النتائج
                results["searches"].append({
                    "topic": topic,
                    "mode": search_mode.name,
                    "total_results": search_response.total_results,
                    "search_time": search_response.search_time
                })
                
                # جمع البيانات من نتائج البحث
                sources = [result.url for result in search_response.results]
                
                # إنشاء تكوين جمع البيانات
                for source_type in config.source_types:
                    # إنشاء تكوين جمع البيانات
                    collection_config = DataCollectionConfig(
                        sources=sources,
                        source_type=source_type,
                        expected_format=DataFormat.HTML,
                        max_items=config.max_sources,
                        depth=config.update_depth,
                        follow_links=True,
                        transformations=["html_to_text", "extract_metadata"],
                        storage_path=os.path.join(self.storage_dir, f"{topic}_{source_type.name}_{int(time.time())}.json")
                    )
                    
                    # جمع البيانات
                    collection_result = self.data_collector.collect(collection_config)
                    
                    # إضافة نتيجة جمع البيانات إلى النتائج
                    results["collections"].append({
                        "topic": topic,
                        "source_type": source_type.name,
                        "total_items": collection_result.total_items,
                        "collection_time": collection_result.collection_time,
                        "success_rate": collection_result.success_rate
                    })
                    
                    # تحديث إجمالي المصادر
                    results["total_sources"] += collection_result.total_items
                    
                    # استخراج المفاهيم من البيانات المجمعة
                    concepts = set()
                    for item in collection_result.items:
                        # استخراج النص
                        text = ""
                        
                        if item.format == DataFormat.TEXT:
                            text = item.content
                        elif item.format == DataFormat.HTML:
                            # تحويل HTML إلى نص
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(item.content, 'html.parser')
                            text = soup.get_text()
                        elif item.format == DataFormat.JSON:
                            # تحويل JSON إلى نص
                            if isinstance(item.content, dict) or isinstance(item.content, list):
                                text = json.dumps(item.content, ensure_ascii=False)
                            else:
                                text = str(item.content)
                        
                        # استخراج المفاهيم
                        if text:
                            extracted_concepts = self.knowledge_extractor.extract_concepts(text)
                            concepts.update(extracted_concepts)
                    
                    # تحديث الرسم البياني المفاهيمي
                    for concept_name in concepts:
                        # التحقق من وجود المفهوم
                        concept_exists = False
                        concept_node = None
                        
                        for node_id, node in self.conceptual_graph.nodes.items():
                            if node.name.lower() == concept_name.lower():
                                concept_exists = True
                                concept_node = node
                                break
                        
                        if concept_exists and config.update_existing:
                            # تحديث المفهوم الموجود
                            # استخراج المتجه الدلالي للمفهوم
                            semantic_vector = self.knowledge_extractor.extract_semantic_vector(concept_name)
                            
                            # تحديث المتجه الدلالي
                            if concept_node.semantic_vector:
                                # دمج المتجهين
                                concept_node.semantic_vector = concept_node.semantic_vector.blend(semantic_vector)
                            else:
                                concept_node.semantic_vector = semantic_vector
                            
                            # تحديث عدد المفاهيم المحدثة
                            results["concepts_updated"] += 1
                        
                        elif not concept_exists and config.add_new_concepts:
                            # إضافة المفهوم الجديد
                            # إنشاء معرف فريد للمفهوم
                            concept_id = f"concept_{len(self.conceptual_graph.nodes) + 1}"
                            
                            # استخراج المتجه الدلالي للمفهوم
                            semantic_vector = self.knowledge_extractor.extract_semantic_vector(concept_name)
                            
                            # إنشاء عقدة المفهوم
                            node = ConceptNode(
                                id=concept_id,
                                name=concept_name,
                                description=f"مفهوم مستخرج من البحث عن: {topic}",
                                semantic_vector=semantic_vector
                            )
                            
                            # إضافة المفهوم إلى الرسم البياني
                            self.conceptual_graph.add_node(node)
                            
                            # تحديث عدد المفاهيم المضافة
                            results["concepts_added"] += 1
                    
                    # إضافة العلاقات بين المفاهيم
                    concepts_list = list(concepts)
                    for i, concept1 in enumerate(concepts_list):
                        for j, concept2 in enumerate(concepts_list):
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
                                        
                                        # تحديث عدد العلاقات المضافة
                                        results["relations_added"] += 1
        
        return results
    
    def schedule_update(self, config: KnowledgeUpdateConfig, schedule_time: Optional[datetime] = None) -> KnowledgeUpdateTask:
        """
        جدولة تحديث المعرفة.
        
        Args:
            config: تكوين التحديث
            schedule_time: وقت الجدولة (اختياري)
            
        Returns:
            مهمة التحديث
        """
        # إنشاء معرف فريد للمهمة
        task_id = f"update_{int(time.time())}_{len(self.tasks) + 1}"
        
        # إنشاء المهمة
        task = KnowledgeUpdateTask(
            config=config,
            task_id=task_id,
            created_at=datetime.now(),
            scheduled_at=schedule_time
        )
        
        # إضافة المهمة إلى قائمة المهام
        self.tasks[task_id] = task
        
        # إضافة المهمة إلى قائمة المهام المجدولة أو قائمة الانتظار
        if schedule_time:
            self.scheduled_tasks.append(task)
        else:
            self.task_queue.put(task)
        
        self.logger.info(f"تمت جدولة مهمة التحديث {task_id}")
        
        return task
    
    def get_task(self, task_id: str) -> Optional[KnowledgeUpdateTask]:
        """
        الحصول على مهمة تحديث المعرفة.
        
        Args:
            task_id: معرف المهمة
            
        Returns:
            مهمة التحديث إذا وجدت، وإلا None
        """
        return self.tasks.get(task_id)
    
    def get_tasks(self) -> List[KnowledgeUpdateTask]:
        """
        الحصول على جميع مهام تحديث المعرفة.
        
        Returns:
            قائمة بمهام التحديث
        """
        return list(self.tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        إلغاء مهمة تحديث المعرفة.
        
        Args:
            task_id: معرف المهمة
            
        Returns:
            True إذا تم إلغاء المهمة، وإلا False
        """
        # التحقق من وجود المهمة
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # إلغاء المهمة إذا كانت مجدولة
        if task in self.scheduled_tasks:
            self.scheduled_tasks.remove(task)
        
        # تحديث حالة المهمة
        task.status = "cancelled"
        
        self.logger.info(f"تم إلغاء مهمة التحديث {task_id}")
        
        return True
    
    def save_state(self, path: str) -> None:
        """
        حفظ حالة مدير تحديث المعرفة.
        
        Args:
            path: مسار الحفظ
        """
        # إنشاء قاموس الحالة
        state = {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "scheduled_tasks": [task.task_id for task in self.scheduled_tasks],
            "storage_dir": self.storage_dir
        }
        
        # حفظ الحالة
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"تم حفظ حالة مدير تحديث المعرفة في {path}")
    
    def load_state(self, path: str) -> None:
        """
        تحميل حالة مدير تحديث المعرفة.
        
        Args:
            path: مسار التحميل
        """
        # تحميل الحالة
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # تحديث مجلد التخزين
        self.storage_dir = state.get("storage_dir", self.storage_dir)
        
        # إنشاء مجلد التخزين إذا لم يكن موجوداً
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # تحديث المهام
        self.tasks = {}
        for task_id, task_dict in state.get("tasks", {}).items():
            # إنشاء تكوين التحديث
            config = KnowledgeUpdateConfig(
                topics=task_dict.get("topics", []),
                frequency=UpdateFrequency[task_dict.get("frequency", UpdateFrequency.DAILY.name)],
                priority=UpdatePriority[task_dict.get("priority", UpdatePriority.MEDIUM.name)],
                max_sources=task_dict.get("max_sources", 10),
                max_concepts=task_dict.get("max_concepts", 100),
                update_depth=task_dict.get("update_depth", 2)
            )
            
            # إنشاء المهمة
            task = KnowledgeUpdateTask(
                config=config,
                task_id=task_id,
                created_at=datetime.fromisoformat(task_dict.get("created_at", datetime.now().isoformat())),
                status=task_dict.get("status", "pending"),
                progress=task_dict.get("progress", 0.0),
                result=task_dict.get("result"),
                error=task_dict.get("error")
            )
            
            # تحديث تواريخ المهمة
            if task_dict.get("scheduled_at"):
                task.scheduled_at = datetime.fromisoformat(task_dict["scheduled_at"])
            
            if task_dict.get("started_at"):
                task.started_at = datetime.fromisoformat(task_dict["started_at"])
            
            if task_dict.get("completed_at"):
                task.completed_at = datetime.fromisoformat(task_dict["completed_at"])
            
            # إضافة المهمة إلى قائمة المهام
            self.tasks[task_id] = task
        
        # تحديث المهام المجدولة
        self.scheduled_tasks = []
        for task_id in state.get("scheduled_tasks", []):
            task = self.tasks.get(task_id)
            if task:
                self.scheduled_tasks.append(task)
        
        self.logger.info(f"تم تحميل حالة مدير تحديث المعرفة من {path}")


class UserFeedbackManager:
    """مدير التغذية الراجعة من المستخدم."""
    
    def __init__(self, storage_dir: str = "user_feedback"):
        """
        تهيئة مدير التغذية الراجعة من المستخدم.
        
        Args:
            storage_dir: مجلد تخزين التغذية الراجعة
        """
        self.logger = logging.getLogger('basira_knowledge_update.user_feedback')
        self.storage_dir = storage_dir
        
        # إنشاء مجلد التخزين إذا لم يكن موجوداً
        os.makedirs(storage_dir, exist_ok=True)
        
        # تهيئة مكونات النظام
        self.architecture = CognitiveLinguisticArchitecture()
        self.knowledge_extractor = KnowledgeExtractor()
        self.knowledge_generator = KnowledgeGenerator()
        self.conceptual_graph = ConceptualGraph()
        
        # تهيئة قائمة التغذية الراجعة
        self.feedback: List[Dict[str, Any]] = []
    
    def add_feedback(self, user_id: str, content: str, rating: Optional[int] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        إضافة تغذية راجعة من المستخدم.
        
        Args:
            user_id: معرف المستخدم
            content: محتوى التغذية الراجعة
            rating: تقييم التغذية الراجعة (اختياري)
            context: سياق التغذية الراجعة (اختياري)
            
        Returns:
            التغذية الراجعة المضافة
        """
        # إنشاء التغذية الراجعة
        feedback = {
            "id": f"feedback_{int(time.time())}_{len(self.feedback) + 1}",
            "user_id": user_id,
            "content": content,
            "rating": rating,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }
        
        # إضافة التغذية الراجعة إلى القائمة
        self.feedback.append(feedback)
        
        # حفظ التغذية الراجعة
        self._save_feedback(feedback)
        
        self.logger.info(f"تمت إضافة تغذية راجعة جديدة: {feedback['id']}")
        
        return feedback
    
    def _save_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        حفظ التغذية الراجعة.
        
        Args:
            feedback: التغذية الراجعة
        """
        # إنشاء مسار الحفظ
        path = os.path.join(self.storage_dir, f"{feedback['id']}.json")
        
        # حفظ التغذية الراجعة
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)
    
    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        الحصول على تغذية راجعة.
        
        Args:
            feedback_id: معرف التغذية الراجعة
            
        Returns:
            التغذية الراجعة إذا وجدت، وإلا None
        """
        for feedback in self.feedback:
            if feedback["id"] == feedback_id:
                return feedback
        
        return None
    
    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """
        الحصول على جميع التغذية الراجعة.
        
        Returns:
            قائمة بالتغذية الراجعة
        """
        return self.feedback
    
    def process_feedback(self, feedback_id: Optional[str] = None) -> Dict[str, Any]:
        """
        معالجة التغذية الراجعة.
        
        Args:
            feedback_id: معرف التغذية الراجعة (اختياري)
            
        Returns:
            نتيجة المعالجة
        """
        results = {
            "processed": 0,
            "concepts_added": 0,
            "concepts_updated": 0,
            "relations_added": 0
        }
        
        # تحديد التغذية الراجعة المراد معالجتها
        feedback_to_process = []
        
        if feedback_id:
            # معالجة تغذية راجعة محددة
            feedback = self.get_feedback(feedback_id)
            if feedback:
                feedback_to_process.append(feedback)
        else:
            # معالجة جميع التغذية الراجعة غير المعالجة
            feedback_to_process = [f for f in self.feedback if not f["processed"]]
        
        # معالجة التغذية الراجعة
        for feedback in feedback_to_process:
            try:
                # استخراج المفاهيم من التغذية الراجعة
                concepts = self.knowledge_extractor.extract_concepts(feedback["content"])
                
                # تحديث الرسم البياني المفاهيمي
                for concept_name in concepts:
                    # التحقق من وجود المفهوم
                    concept_exists = False
                    concept_node = None
                    
                    for node_id, node in self.conceptual_graph.nodes.items():
                        if node.name.lower() == concept_name.lower():
                            concept_exists = True
                            concept_node = node
                            break
                    
                    if concept_exists:
                        # تحديث المفهوم الموجود
                        # استخراج المتجه الدلالي للمفهوم
                        semantic_vector = self.knowledge_extractor.extract_semantic_vector(concept_name)
                        
                        # تحديث المتجه الدلالي
                        if concept_node.semantic_vector:
                            # دمج المتجهين
                            concept_node.semantic_vector = concept_node.semantic_vector.blend(semantic_vector)
                        else:
                            concept_node.semantic_vector = semantic_vector
                        
                        # تحديث عدد المفاهيم المحدثة
                        results["concepts_updated"] += 1
                    else:
                        # إضافة المفهوم الجديد
                        # إنشاء معرف فريد للمفهوم
                        concept_id = f"concept_{len(self.conceptual_graph.nodes) + 1}"
                        
                        # استخراج المتجه الدلالي للمفهوم
                        semantic_vector = self.knowledge_extractor.extract_semantic_vector(concept_name)
                        
                        # إنشاء عقدة المفهوم
                        node = ConceptNode(
                            id=concept_id,
                            name=concept_name,
                            description=f"مفهوم مستخرج من تغذية راجعة المستخدم",
                            semantic_vector=semantic_vector
                        )
                        
                        # إضافة المفهوم إلى الرسم البياني
                        self.conceptual_graph.add_node(node)
                        
                        # تحديث عدد المفاهيم المضافة
                        results["concepts_added"] += 1
                
                # إضافة العلاقات بين المفاهيم
                concepts_list = list(concepts)
                for i, concept1 in enumerate(concepts_list):
                    for j, concept2 in enumerate(concepts_list):
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
                                    
                                    # تحديث عدد العلاقات المضافة
                                    results["relations_added"] += 1
                
                # تحديث حالة التغذية الراجعة
                feedback["processed"] = True
                
                # حفظ التغذية الراجعة
                self._save_feedback(feedback)
                
                # تحديث عدد التغذية الراجعة المعالجة
                results["processed"] += 1
            
            except Exception as e:
                self.logger.error(f"فشل في معالجة التغذية الراجعة {feedback['id']}: {e}")
        
        return results
    
    def load_feedback(self) -> None:
        """تحميل التغذية الراجعة من الملفات."""
        # مسح قائمة التغذية الراجعة
        self.feedback = []
        
        # تحميل الملفات
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.storage_dir, filename)
                
                try:
                    # تحميل التغذية الراجعة
                    with open(path, 'r', encoding='utf-8') as f:
                        feedback = json.load(f)
                    
                    # إضافة التغذية الراجعة إلى القائمة
                    self.feedback.append(feedback)
                
                except Exception as e:
                    self.logger.error(f"فشل في تحميل التغذية الراجعة {filename}: {e}")
        
        self.logger.info(f"تم تحميل {len(self.feedback)} تغذية راجعة")


# تنفيذ الاختبار إذا تم تشغيل الملف مباشرة
if __name__ == "__main__":
    # إنشاء مدير تحديث المعرفة
    update_manager = KnowledgeUpdateManager()
    
    # إنشاء تكوين تحديث المعرفة
    config = KnowledgeUpdateConfig(
        topics=["الذكاء الاصطناعي", "معالجة اللغة الطبيعية"],
        frequency=UpdateFrequency.DAILY,
        priority=UpdatePriority.MEDIUM,
        max_sources=5,
        max_concepts=50,
        update_depth=1
    )
    
    # جدولة تحديث المعرفة
    task = update_manager.schedule_update(config)
    
    # بدء مدير تحديث المعرفة
    update_manager.start()
    
    # انتظار إكمال المهمة
    while task.status != "completed" and task.status != "failed":
        print(f"حالة المهمة: {task.status}, التقدم: {task.progress:.2f}")
        time.sleep(5)
    
    # عرض نتيجة المهمة
    if task.status == "completed":
        print("تم إكمال المهمة بنجاح")
        print(f"تمت إضافة {task.result['concepts_added']} مفهوم جديد")
        print(f"تم تحديث {task.result['concepts_updated']} مفهوم موجود")
        print(f"تمت إضافة {task.result['relations_added']} علاقة جديدة")
    else:
        print(f"فشل في إكمال المهمة: {task.error}")
    
    # إيقاف مدير تحديث المعرفة
    update_manager.stop()
    
    # إنشاء مدير التغذية الراجعة من المستخدم
    feedback_manager = UserFeedbackManager()
    
    # إضافة تغذية راجعة
    feedback = feedback_manager.add_feedback(
        user_id="user1",
        content="أعتقد أن الذكاء الاصطناعي سيغير مستقبل البشرية بشكل كبير. يجب أن نركز على تطوير الذكاء الاصطناعي الآمن والأخلاقي.",
        rating=5,
        context={"source": "chat", "topic": "الذكاء الاصطناعي"}
    )
    
    # معالجة التغذية الراجعة
    results = feedback_manager.process_feedback(feedback["id"])
    
    # عرض نتائج المعالجة
    print(f"تمت معالجة {results['processed']} تغذية راجعة")
    print(f"تمت إضافة {results['concepts_added']} مفهوم جديد")
    print(f"تم تحديث {results['concepts_updated']} مفهوم موجود")
    print(f"تمت إضافة {results['relations_added']} علاقة جديدة")
