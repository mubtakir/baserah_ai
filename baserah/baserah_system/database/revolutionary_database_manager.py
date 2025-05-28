#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مدير قواعد البيانات الثوري - Revolutionary Database Manager
نظام شامل لإدارة وحفظ المعرفة المكتسبة في جميع وحدات النظام الثوري

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - Revolutionary Database System
"""

import os
import json
import sqlite3
import pickle
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """إدخال معرفة في قاعدة البيانات"""
    id: str
    module_name: str
    knowledge_type: str
    content: Dict[str, Any]
    confidence_level: float
    timestamp: str
    metadata: Dict[str, Any]
    ai_oop_applied: bool = True
    revolutionary_method: str = "expert_explorer"


class RevolutionaryDatabaseManager:
    """
    مدير قواعد البيانات الثوري
    يدير جميع قواعد البيانات للوحدات المختلفة
    """
    
    def __init__(self, base_path: str = "database"):
        """تهيئة مدير قواعد البيانات"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # قواعد البيانات المختلفة
        self.databases = {}
        self.connections = {}
        
        # خيط الحفظ التلقائي
        self.auto_save_thread = None
        self.auto_save_interval = 300  # 5 دقائق
        self.running = True
        
        # قفل للأمان في البيئة متعددة الخيوط
        self.lock = threading.Lock()
        
        # تهيئة قواعد البيانات
        self._initialize_databases()
        
        # بدء الحفظ التلقائي
        self._start_auto_save()
        
        logger.info("🗄️ تم تهيئة مدير قواعد البيانات الثوري")
    
    def _initialize_databases(self):
        """تهيئة جميع قواعد البيانات المطلوبة"""
        
        # قواعد البيانات للوحدات المختلفة
        database_configs = {
            "revolutionary_learning": {
                "type": "sqlite",
                "file": "revolutionary_learning.db",
                "tables": ["expert_decisions", "exploration_results", "wisdom_accumulation"]
            },
            "adaptive_equations": {
                "type": "sqlite", 
                "file": "adaptive_equations.db",
                "tables": ["equation_patterns", "adaptation_history", "performance_metrics"]
            },
            "dream_interpretation": {
                "type": "sqlite",
                "file": "dream_interpretation.db", 
                "tables": ["dream_symbols", "interpretation_history", "user_feedback"]
            },
            "arabic_nlp": {
                "type": "json",
                "file": "arabic_nlp_knowledge.json",
                "structure": {"roots": {}, "semantics": {}, "patterns": {}}
            },
            "mathematical_core": {
                "type": "sqlite",
                "file": "mathematical_knowledge.db",
                "tables": ["equations", "solutions", "innovations"]
            },
            "system_integration": {
                "type": "json",
                "file": "system_integration.json",
                "structure": {"connections": {}, "performance": {}, "evolution": {}}
            },
            "user_interactions": {
                "type": "sqlite",
                "file": "user_interactions.db",
                "tables": ["sessions", "feedback", "preferences"]
            },
            "revolutionary_wisdom": {
                "type": "json",
                "file": "revolutionary_wisdom.json",
                "structure": {"insights": {}, "patterns": {}, "evolution": {}}
            }
        }
        
        for db_name, config in database_configs.items():
            self._create_database(db_name, config)
    
    def _create_database(self, name: str, config: Dict[str, Any]):
        """إنشاء قاعدة بيانات محددة"""
        db_path = self.base_path / config["file"]
        
        if config["type"] == "sqlite":
            self._create_sqlite_database(name, db_path, config.get("tables", []))
        elif config["type"] == "json":
            self._create_json_database(name, db_path, config.get("structure", {}))
        
        self.databases[name] = {
            "type": config["type"],
            "path": db_path,
            "config": config
        }
        
        logger.info(f"✅ تم إنشاء قاعدة بيانات {name}")
    
    def _create_sqlite_database(self, name: str, path: Path, tables: List[str]):
        """إنشاء قاعدة بيانات SQLite"""
        conn = sqlite3.connect(str(path))
        
        # إنشاء جدول المعرفة العام
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                module_name TEXT,
                knowledge_type TEXT,
                content TEXT,
                confidence_level REAL,
                timestamp TEXT,
                metadata TEXT,
                ai_oop_applied BOOLEAN,
                revolutionary_method TEXT
            )
        """)
        
        # إنشاء الجداول المخصصة
        for table in tables:
            if table == "expert_decisions":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS expert_decisions (
                        id TEXT PRIMARY KEY,
                        situation TEXT,
                        decision TEXT,
                        confidence REAL,
                        reasoning TEXT,
                        timestamp TEXT
                    )
                """)
            elif table == "exploration_results":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS exploration_results (
                        id TEXT PRIMARY KEY,
                        exploration_type TEXT,
                        discoveries TEXT,
                        novelty_score REAL,
                        timestamp TEXT
                    )
                """)
            elif table == "wisdom_accumulation":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS wisdom_accumulation (
                        id TEXT PRIMARY KEY,
                        wisdom_type TEXT,
                        wisdom_content TEXT,
                        accumulation_level REAL,
                        timestamp TEXT
                    )
                """)
            elif table == "equation_patterns":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS equation_patterns (
                        id TEXT PRIMARY KEY,
                        pattern_type TEXT,
                        equation TEXT,
                        performance REAL,
                        adaptation_count INTEGER,
                        timestamp TEXT
                    )
                """)
            elif table == "dream_symbols":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dream_symbols (
                        id TEXT PRIMARY KEY,
                        symbol TEXT,
                        meaning TEXT,
                        context TEXT,
                        confidence REAL,
                        timestamp TEXT
                    )
                """)
        
        conn.commit()
        conn.close()
    
    def _create_json_database(self, name: str, path: Path, structure: Dict[str, Any]):
        """إنشاء قاعدة بيانات JSON"""
        if not path.exists():
            initial_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "3.0.0",
                    "type": "revolutionary_knowledge",
                    "ai_oop_applied": True
                },
                "data": structure
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    def save_knowledge(self, module_name: str, knowledge_type: str, 
                      content: Dict[str, Any], confidence_level: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        حفظ معرفة جديدة في قاعدة البيانات
        
        Args:
            module_name: اسم الوحدة
            knowledge_type: نوع المعرفة
            content: محتوى المعرفة
            confidence_level: مستوى الثقة
            metadata: بيانات إضافية
        
        Returns:
            معرف الإدخال المحفوظ
        """
        with self.lock:
            # إنشاء إدخال المعرفة
            entry_id = f"{module_name}_{knowledge_type}_{int(time.time())}"
            
            entry = KnowledgeEntry(
                id=entry_id,
                module_name=module_name,
                knowledge_type=knowledge_type,
                content=content,
                confidence_level=confidence_level,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {},
                ai_oop_applied=True,
                revolutionary_method="expert_explorer"
            )
            
            # حفظ في قاعدة البيانات المناسبة
            self._save_to_appropriate_database(entry)
            
            logger.info(f"💾 تم حفظ معرفة جديدة: {entry_id}")
            return entry_id
    
    def _save_to_appropriate_database(self, entry: KnowledgeEntry):
        """حفظ الإدخال في قاعدة البيانات المناسبة"""
        
        # تحديد قاعدة البيانات المناسبة
        if entry.module_name in ["revolutionary_learning", "expert_explorer"]:
            db_name = "revolutionary_learning"
        elif entry.module_name in ["adaptive_equations", "mathematical"]:
            db_name = "adaptive_equations" if "equation" in entry.knowledge_type else "mathematical_core"
        elif entry.module_name in ["dream_interpretation", "dream"]:
            db_name = "dream_interpretation"
        elif entry.module_name in ["arabic_nlp", "arabic"]:
            db_name = "arabic_nlp"
        else:
            db_name = "system_integration"
        
        if db_name not in self.databases:
            logger.warning(f"⚠️ قاعدة البيانات {db_name} غير موجودة")
            return
        
        db_config = self.databases[db_name]
        
        if db_config["type"] == "sqlite":
            self._save_to_sqlite(db_config["path"], entry)
        elif db_config["type"] == "json":
            self._save_to_json(db_config["path"], entry)
    
    def _save_to_sqlite(self, db_path: Path, entry: KnowledgeEntry):
        """حفظ في قاعدة بيانات SQLite"""
        conn = sqlite3.connect(str(db_path))
        
        conn.execute("""
            INSERT OR REPLACE INTO knowledge_entries 
            (id, module_name, knowledge_type, content, confidence_level, 
             timestamp, metadata, ai_oop_applied, revolutionary_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.module_name,
            entry.knowledge_type,
            json.dumps(entry.content, ensure_ascii=False),
            entry.confidence_level,
            entry.timestamp,
            json.dumps(entry.metadata, ensure_ascii=False),
            entry.ai_oop_applied,
            entry.revolutionary_method
        ))
        
        conn.commit()
        conn.close()
    
    def _save_to_json(self, db_path: Path, entry: KnowledgeEntry):
        """حفظ في قاعدة بيانات JSON"""
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"metadata": {}, "data": {}}
        
        # إضافة الإدخال الجديد
        if "entries" not in data["data"]:
            data["data"]["entries"] = {}
        
        data["data"]["entries"][entry.id] = asdict(entry)
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_knowledge(self, module_name: str, knowledge_type: Optional[str] = None,
                      limit: int = 100) -> List[KnowledgeEntry]:
        """
        تحميل المعرفة من قاعدة البيانات
        
        Args:
            module_name: اسم الوحدة
            knowledge_type: نوع المعرفة (اختياري)
            limit: حد عدد النتائج
        
        Returns:
            قائمة بإدخالات المعرفة
        """
        with self.lock:
            entries = []
            
            # البحث في جميع قواعد البيانات
            for db_name, db_config in self.databases.items():
                if db_config["type"] == "sqlite":
                    entries.extend(self._load_from_sqlite(
                        db_config["path"], module_name, knowledge_type, limit
                    ))
                elif db_config["type"] == "json":
                    entries.extend(self._load_from_json(
                        db_config["path"], module_name, knowledge_type, limit
                    ))
            
            # ترتيب حسب الوقت (الأحدث أولاً)
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            return entries[:limit]
    
    def _load_from_sqlite(self, db_path: Path, module_name: str, 
                         knowledge_type: Optional[str], limit: int) -> List[KnowledgeEntry]:
        """تحميل من قاعدة بيانات SQLite"""
        if not db_path.exists():
            return []
        
        conn = sqlite3.connect(str(db_path))
        
        query = "SELECT * FROM knowledge_entries WHERE module_name = ?"
        params = [module_name]
        
        if knowledge_type:
            query += " AND knowledge_type = ?"
            params.append(knowledge_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        entries = []
        for row in rows:
            try:
                entry = KnowledgeEntry(
                    id=row[0],
                    module_name=row[1],
                    knowledge_type=row[2],
                    content=json.loads(row[3]),
                    confidence_level=row[4],
                    timestamp=row[5],
                    metadata=json.loads(row[6]),
                    ai_oop_applied=bool(row[7]),
                    revolutionary_method=row[8]
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(f"⚠️ خطأ في تحميل إدخال: {e}")
        
        return entries
    
    def _load_from_json(self, db_path: Path, module_name: str,
                       knowledge_type: Optional[str], limit: int) -> List[KnowledgeEntry]:
        """تحميل من قاعدة بيانات JSON"""
        if not db_path.exists():
            return []
        
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        
        entries = []
        entries_data = data.get("data", {}).get("entries", {})
        
        for entry_id, entry_data in entries_data.items():
            if entry_data.get("module_name") == module_name:
                if not knowledge_type or entry_data.get("knowledge_type") == knowledge_type:
                    try:
                        entry = KnowledgeEntry(**entry_data)
                        entries.append(entry)
                    except Exception as e:
                        logger.warning(f"⚠️ خطأ في تحميل إدخال JSON: {e}")
        
        return entries[:limit]
    
    def _start_auto_save(self):
        """بدء الحفظ التلقائي"""
        def auto_save_worker():
            while self.running:
                time.sleep(self.auto_save_interval)
                if self.running:
                    self._backup_databases()
        
        self.auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_thread.start()
        
        logger.info(f"🔄 تم بدء الحفظ التلقائي كل {self.auto_save_interval} ثانية")
    
    def _backup_databases(self):
        """إنشاء نسخ احتياطية من قواعد البيانات"""
        backup_dir = self.base_path / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for db_name, db_config in self.databases.items():
            try:
                source = db_config["path"]
                if source.exists():
                    destination = backup_dir / source.name
                    
                    if db_config["type"] == "sqlite":
                        # نسخ قاعدة بيانات SQLite
                        import shutil
                        shutil.copy2(source, destination)
                    elif db_config["type"] == "json":
                        # نسخ ملف JSON
                        import shutil
                        shutil.copy2(source, destination)
                        
            except Exception as e:
                logger.warning(f"⚠️ فشل في نسخ قاعدة البيانات {db_name}: {e}")
        
        logger.info(f"💾 تم إنشاء نسخة احتياطية في {backup_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات قواعد البيانات"""
        stats = {
            "total_databases": len(self.databases),
            "database_details": {},
            "total_entries": 0,
            "last_backup": None
        }
        
        for db_name, db_config in self.databases.items():
            db_stats = {
                "type": db_config["type"],
                "size": 0,
                "entries": 0
            }
            
            if db_config["path"].exists():
                db_stats["size"] = db_config["path"].stat().st_size
                
                if db_config["type"] == "sqlite":
                    try:
                        conn = sqlite3.connect(str(db_config["path"]))
                        cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
                        db_stats["entries"] = cursor.fetchone()[0]
                        conn.close()
                    except Exception:
                        pass
                elif db_config["type"] == "json":
                    try:
                        with open(db_config["path"], 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        db_stats["entries"] = len(data.get("data", {}).get("entries", {}))
                    except Exception:
                        pass
            
            stats["database_details"][db_name] = db_stats
            stats["total_entries"] += db_stats["entries"]
        
        return stats
    
    def close(self):
        """إغلاق مدير قواعد البيانات"""
        self.running = False
        
        if self.auto_save_thread and self.auto_save_thread.is_alive():
            self.auto_save_thread.join(timeout=5)
        
        # إنشاء نسخة احتياطية أخيرة
        self._backup_databases()
        
        logger.info("🔒 تم إغلاق مدير قواعد البيانات الثوري")


# إنشاء مثيل عام لمدير قواعد البيانات
_global_db_manager = None

def get_database_manager() -> RevolutionaryDatabaseManager:
    """الحصول على مدير قواعد البيانات العام"""
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = RevolutionaryDatabaseManager()
    return _global_db_manager


if __name__ == "__main__":
    # اختبار مدير قواعد البيانات
    db_manager = RevolutionaryDatabaseManager()
    
    # حفظ معرفة تجريبية
    knowledge_id = db_manager.save_knowledge(
        module_name="revolutionary_learning",
        knowledge_type="expert_decision",
        content={
            "situation": "test_situation",
            "decision": "test_decision",
            "confidence": 0.95
        },
        confidence_level=0.95,
        metadata={"test": True}
    )
    
    print(f"✅ تم حفظ المعرفة: {knowledge_id}")
    
    # تحميل المعرفة
    loaded_knowledge = db_manager.load_knowledge("revolutionary_learning")
    print(f"📚 تم تحميل {len(loaded_knowledge)} إدخال معرفة")
    
    # عرض الإحصائيات
    stats = db_manager.get_statistics()
    print(f"📊 إحصائيات قواعد البيانات: {stats}")
    
    # إغلاق المدير
    db_manager.close()
