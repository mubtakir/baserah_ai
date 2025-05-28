#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
خليط حفظ المعرفة - Knowledge Persistence Mixin
يضاف إلى جميع الوحدات لحفظ المعرفة المكتسبة تلقائياً

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - Revolutionary Knowledge Persistence
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging

# استيراد مدير قواعد البيانات
try:
    from .revolutionary_database_manager import get_database_manager
except ImportError:
    from database.revolutionary_database_manager import get_database_manager

logger = logging.getLogger(__name__)


class KnowledgePersistenceMixin:
    """
    خليط حفظ المعرفة - يضاف إلى جميع الوحدات الثورية
    يحفظ المعرفة المكتسبة تلقائياً ويستعيدها عند الحاجة
    """
    
    def __init__(self, *args, **kwargs):
        """تهيئة نظام حفظ المعرفة"""
        super().__init__(*args, **kwargs)
        
        # مدير قواعد البيانات
        self.db_manager = get_database_manager()
        
        # معلومات الوحدة
        self.module_name = getattr(self, 'module_name', self.__class__.__name__)
        
        # ذاكرة المعرفة المحلية
        self.local_knowledge_cache = {}
        
        # إعدادات الحفظ
        self.auto_save_enabled = True
        self.save_threshold = 10  # حفظ كل 10 إدخالات جديدة
        self.unsaved_count = 0
        
        # تحميل المعرفة المحفوظة
        self._load_existing_knowledge()
        
        logger.info(f"🧠 تم تهيئة نظام حفظ المعرفة للوحدة: {self.module_name}")
    
    def _load_existing_knowledge(self):
        """تحميل المعرفة المحفوظة مسبقاً"""
        try:
            # تحميل جميع أنواع المعرفة للوحدة
            knowledge_entries = self.db_manager.load_knowledge(
                module_name=self.module_name,
                limit=1000  # تحميل آخر 1000 إدخال
            )
            
            # تنظيم المعرفة حسب النوع
            for entry in knowledge_entries:
                knowledge_type = entry.knowledge_type
                
                if knowledge_type not in self.local_knowledge_cache:
                    self.local_knowledge_cache[knowledge_type] = []
                
                self.local_knowledge_cache[knowledge_type].append({
                    "content": entry.content,
                    "confidence": entry.confidence_level,
                    "timestamp": entry.timestamp,
                    "metadata": entry.metadata
                })
            
            total_loaded = sum(len(entries) for entries in self.local_knowledge_cache.values())
            logger.info(f"📚 تم تحميل {total_loaded} إدخال معرفة للوحدة {self.module_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ فشل في تحميل المعرفة للوحدة {self.module_name}: {e}")
    
    def save_knowledge(self, knowledge_type: str, content: Dict[str, Any],
                      confidence_level: float = 1.0, metadata: Optional[Dict[str, Any]] = None,
                      force_save: bool = False) -> str:
        """
        حفظ معرفة جديدة
        
        Args:
            knowledge_type: نوع المعرفة
            content: محتوى المعرفة
            confidence_level: مستوى الثقة
            metadata: بيانات إضافية
            force_save: إجبار الحفظ الفوري
        
        Returns:
            معرف المعرفة المحفوظة
        """
        # إضافة إلى الذاكرة المحلية
        if knowledge_type not in self.local_knowledge_cache:
            self.local_knowledge_cache[knowledge_type] = []
        
        knowledge_entry = {
            "content": content,
            "confidence": confidence_level,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.local_knowledge_cache[knowledge_type].append(knowledge_entry)
        
        # حفظ في قاعدة البيانات
        knowledge_id = self.db_manager.save_knowledge(
            module_name=self.module_name,
            knowledge_type=knowledge_type,
            content=content,
            confidence_level=confidence_level,
            metadata=metadata
        )
        
        # تحديث عداد الحفظ
        self.unsaved_count += 1
        
        # حفظ تلقائي إذا وصل العتبة
        if self.auto_save_enabled and (self.unsaved_count >= self.save_threshold or force_save):
            self._trigger_auto_save()
        
        logger.debug(f"💾 تم حفظ معرفة {knowledge_type} للوحدة {self.module_name}")
        return knowledge_id
    
    def load_knowledge(self, knowledge_type: str, limit: int = 100,
                      confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        تحميل معرفة محددة
        
        Args:
            knowledge_type: نوع المعرفة
            limit: حد عدد النتائج
            confidence_threshold: حد الثقة الأدنى
        
        Returns:
            قائمة بإدخالات المعرفة
        """
        # البحث في الذاكرة المحلية أولاً
        local_entries = self.local_knowledge_cache.get(knowledge_type, [])
        
        # تصفية حسب مستوى الثقة
        filtered_entries = [
            entry for entry in local_entries
            if entry["confidence"] >= confidence_threshold
        ]
        
        # ترتيب حسب الوقت (الأحدث أولاً)
        filtered_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return filtered_entries[:limit]
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص المعرفة المحفوظة"""
        summary = {
            "module_name": self.module_name,
            "total_knowledge_types": len(self.local_knowledge_cache),
            "knowledge_breakdown": {},
            "total_entries": 0,
            "average_confidence": 0.0,
            "last_updated": None
        }
        
        total_confidence = 0.0
        total_entries = 0
        latest_timestamp = None
        
        for knowledge_type, entries in self.local_knowledge_cache.items():
            type_summary = {
                "count": len(entries),
                "average_confidence": 0.0,
                "latest_entry": None
            }
            
            if entries:
                # حساب متوسط الثقة
                confidences = [entry["confidence"] for entry in entries]
                type_summary["average_confidence"] = sum(confidences) / len(confidences)
                
                # آخر إدخال
                latest_entry = max(entries, key=lambda x: x["timestamp"])
                type_summary["latest_entry"] = latest_entry["timestamp"]
                
                # تحديث الإحصائيات العامة
                total_confidence += sum(confidences)
                total_entries += len(entries)
                
                if latest_timestamp is None or latest_entry["timestamp"] > latest_timestamp:
                    latest_timestamp = latest_entry["timestamp"]
            
            summary["knowledge_breakdown"][knowledge_type] = type_summary
        
        # حساب المتوسطات العامة
        if total_entries > 0:
            summary["average_confidence"] = total_confidence / total_entries
        
        summary["total_entries"] = total_entries
        summary["last_updated"] = latest_timestamp
        
        return summary
    
    def _trigger_auto_save(self):
        """تشغيل الحفظ التلقائي"""
        try:
            # إعادة تعيين العداد
            self.unsaved_count = 0
            
            # يمكن إضافة منطق حفظ إضافي هنا
            logger.debug(f"🔄 تم تشغيل الحفظ التلقائي للوحدة {self.module_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ فشل في الحفظ التلقائي للوحدة {self.module_name}: {e}")
    
    def backup_knowledge(self, backup_path: Optional[str] = None) -> str:
        """
        إنشاء نسخة احتياطية من المعرفة
        
        Args:
            backup_path: مسار النسخة الاحتياطية
        
        Returns:
            مسار النسخة الاحتياطية
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_{self.module_name}_{timestamp}.json"
        
        backup_data = {
            "module_name": self.module_name,
            "backup_timestamp": datetime.now().isoformat(),
            "knowledge_cache": self.local_knowledge_cache,
            "summary": self.get_knowledge_summary()
        }
        
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 تم إنشاء نسخة احتياطية للوحدة {self.module_name}: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"❌ فشل في إنشاء النسخة الاحتياطية للوحدة {self.module_name}: {e}")
            raise
    
    def restore_knowledge(self, backup_path: str) -> bool:
        """
        استعادة المعرفة من نسخة احتياطية
        
        Args:
            backup_path: مسار النسخة الاحتياطية
        
        Returns:
            True إذا نجحت الاستعادة
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # التحقق من صحة البيانات
            if backup_data.get("module_name") != self.module_name:
                logger.warning(f"⚠️ النسخة الاحتياطية لوحدة مختلفة: {backup_data.get('module_name')}")
            
            # استعادة المعرفة
            self.local_knowledge_cache = backup_data.get("knowledge_cache", {})
            
            logger.info(f"✅ تم استعادة المعرفة للوحدة {self.module_name} من {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ فشل في استعادة المعرفة للوحدة {self.module_name}: {e}")
            return False
    
    def clear_knowledge(self, knowledge_type: Optional[str] = None, 
                       confidence_threshold: Optional[float] = None):
        """
        مسح المعرفة
        
        Args:
            knowledge_type: نوع المعرفة المراد مسحها (None لمسح الكل)
            confidence_threshold: مسح المعرفة تحت هذا المستوى من الثقة
        """
        if knowledge_type is None:
            # مسح جميع المعرفة
            if confidence_threshold is None:
                self.local_knowledge_cache.clear()
                logger.info(f"🗑️ تم مسح جميع المعرفة للوحدة {self.module_name}")
            else:
                # مسح المعرفة ضعيفة الثقة
                for k_type in list(self.local_knowledge_cache.keys()):
                    self.local_knowledge_cache[k_type] = [
                        entry for entry in self.local_knowledge_cache[k_type]
                        if entry["confidence"] >= confidence_threshold
                    ]
                logger.info(f"🗑️ تم مسح المعرفة ضعيفة الثقة (<{confidence_threshold}) للوحدة {self.module_name}")
        else:
            # مسح نوع معرفة محدد
            if knowledge_type in self.local_knowledge_cache:
                if confidence_threshold is None:
                    del self.local_knowledge_cache[knowledge_type]
                    logger.info(f"🗑️ تم مسح معرفة {knowledge_type} للوحدة {self.module_name}")
                else:
                    self.local_knowledge_cache[knowledge_type] = [
                        entry for entry in self.local_knowledge_cache[knowledge_type]
                        if entry["confidence"] >= confidence_threshold
                    ]
                    logger.info(f"🗑️ تم مسح معرفة {knowledge_type} ضعيفة الثقة للوحدة {self.module_name}")
    
    def optimize_knowledge_storage(self):
        """تحسين تخزين المعرفة"""
        # إزالة الإدخالات المكررة
        for knowledge_type in self.local_knowledge_cache:
            entries = self.local_knowledge_cache[knowledge_type]
            
            # إزالة المكررات بناءً على المحتوى
            unique_entries = []
            seen_contents = set()
            
            for entry in entries:
                content_hash = hash(json.dumps(entry["content"], sort_keys=True))
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_entries.append(entry)
            
            # الاحتفاظ بأحدث 1000 إدخال فقط
            unique_entries.sort(key=lambda x: x["timestamp"], reverse=True)
            self.local_knowledge_cache[knowledge_type] = unique_entries[:1000]
        
        logger.info(f"🔧 تم تحسين تخزين المعرفة للوحدة {self.module_name}")


class PersistentRevolutionaryComponent(KnowledgePersistenceMixin):
    """
    مكون ثوري دائم - فئة أساسية لجميع المكونات الثورية
    تتضمن نظام حفظ المعرفة تلقائياً
    """
    
    def __init__(self, module_name: str = None, **kwargs):
        """تهيئة المكون الثوري الدائم"""
        self.module_name = module_name or self.__class__.__name__
        super().__init__(**kwargs)
        
        # إعدادات المكون الثوري
        self.ai_oop_applied = True
        self.revolutionary_method = "expert_explorer"
        
        logger.info(f"🌟 تم تهيئة المكون الثوري الدائم: {self.module_name}")
    
    def learn_from_experience(self, experience: Dict[str, Any], 
                            confidence: float = 1.0) -> str:
        """
        التعلم من التجربة وحفظها
        
        Args:
            experience: التجربة المكتسبة
            confidence: مستوى الثقة في التجربة
        
        Returns:
            معرف التجربة المحفوظة
        """
        return self.save_knowledge(
            knowledge_type="experience",
            content=experience,
            confidence_level=confidence,
            metadata={
                "learning_method": "revolutionary_experience",
                "ai_oop_applied": True
            }
        )
    
    def apply_learned_knowledge(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        تطبيق المعرفة المكتسبة على موقف جديد
        
        Args:
            situation: الموقف الحالي
        
        Returns:
            المعرفة المطبقة والقرار
        """
        # البحث عن تجارب مشابهة
        relevant_experiences = self.load_knowledge("experience", limit=50)
        
        # تحليل التشابه وتطبيق المعرفة
        applied_knowledge = {
            "situation": situation,
            "relevant_experiences_count": len(relevant_experiences),
            "confidence": 0.0,
            "decision": None,
            "reasoning": []
        }
        
        if relevant_experiences:
            # تطبيق منطق المطابقة والتعلم
            total_confidence = sum(exp["confidence"] for exp in relevant_experiences)
            applied_knowledge["confidence"] = total_confidence / len(relevant_experiences)
            
            # استخراج القرار من التجارب
            applied_knowledge["decision"] = self._extract_decision_from_experiences(
                situation, relevant_experiences
            )
            
            applied_knowledge["reasoning"] = [
                f"تم تطبيق {len(relevant_experiences)} تجربة مشابهة",
                f"متوسط الثقة: {applied_knowledge['confidence']:.2f}",
                "تم استخدام النظام الثوري الخبير/المستكشف"
            ]
        
        return applied_knowledge
    
    def _extract_decision_from_experiences(self, situation: Dict[str, Any],
                                         experiences: List[Dict[str, Any]]) -> Any:
        """استخراج قرار من التجارب المشابهة"""
        # منطق بسيط لاستخراج القرار
        # يمكن تطويره ليكون أكثر تعقيداً
        
        if not experiences:
            return None
        
        # أخذ القرار من التجربة الأكثر ثقة
        best_experience = max(experiences, key=lambda x: x["confidence"])
        return best_experience["content"].get("decision", "no_decision")


if __name__ == "__main__":
    # اختبار نظام حفظ المعرفة
    
    class TestRevolutionaryComponent(PersistentRevolutionaryComponent):
        def __init__(self):
            super().__init__(module_name="test_component")
    
    # إنشاء مكون تجريبي
    component = TestRevolutionaryComponent()
    
    # حفظ معرفة تجريبية
    knowledge_id = component.save_knowledge(
        knowledge_type="test_knowledge",
        content={"test": "data", "value": 42},
        confidence_level=0.9
    )
    
    print(f"✅ تم حفظ المعرفة: {knowledge_id}")
    
    # تحميل المعرفة
    loaded_knowledge = component.load_knowledge("test_knowledge")
    print(f"📚 تم تحميل {len(loaded_knowledge)} إدخال معرفة")
    
    # عرض ملخص المعرفة
    summary = component.get_knowledge_summary()
    print(f"📊 ملخص المعرفة: {summary}")
    
    # اختبار التعلم من التجربة
    experience_id = component.learn_from_experience({
        "situation": "test_situation",
        "action": "test_action",
        "result": "success",
        "decision": "continue"
    }, confidence=0.95)
    
    print(f"🧠 تم حفظ التجربة: {experience_id}")
    
    # تطبيق المعرفة المكتسبة
    applied = component.apply_learned_knowledge({"new_situation": "similar_test"})
    print(f"🎯 المعرفة المطبقة: {applied}")
