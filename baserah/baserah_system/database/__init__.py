#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام قواعد البيانات الثوري - Revolutionary Database System
تهيئة وإدارة قواعد البيانات لجميع وحدات النظام الثوري

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - Revolutionary Database System
"""

from .revolutionary_database_manager import (
    RevolutionaryDatabaseManager,
    KnowledgeEntry,
    get_database_manager
)

from .knowledge_persistence_mixin import (
    KnowledgePersistenceMixin,
    PersistentRevolutionaryComponent
)

__all__ = [
    'RevolutionaryDatabaseManager',
    'KnowledgeEntry', 
    'get_database_manager',
    'KnowledgePersistenceMixin',
    'PersistentRevolutionaryComponent'
]

__version__ = "3.0.0"
__author__ = "Basil Yahya Abdullah - Iraq/Mosul"
__description__ = "Revolutionary Database System for Basira AI System"
