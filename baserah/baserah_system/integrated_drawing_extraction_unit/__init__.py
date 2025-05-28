#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Drawing-Extraction Unit Package
حزمة الوحدة المتكاملة للرسم والاستنباط

This package contains the integrated unit that combines drawing and extraction
capabilities with an Expert/Explorer bridge for continuous learning and improvement.

هذه الحزمة تحتوي على الوحدة المتكاملة التي تجمع قدرات الرسم والاستنباط
مع جسر الخبير/المستكشف للتعلم والتحسين المستمر.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

from .expert_explorer_bridge import ExpertExplorerBridge, ExpertKnowledge, ExplorerFeedback
from .physics_expert_bridge import PhysicsExpertBridge, PhysicsAnalysisResult, ArtisticPhysicsBalance
from .integrated_unit import IntegratedDrawingExtractionUnit

__all__ = [
    'ExpertExplorerBridge',
    'ExpertKnowledge',
    'ExplorerFeedback',
    'PhysicsExpertBridge',
    'PhysicsAnalysisResult',
    'ArtisticPhysicsBalance',
    'IntegratedDrawingExtractionUnit'
]

__version__ = "1.0.0"
__author__ = "Basil Yahya Abdullah - Iraq/Mosul"
__description__ = "Integrated Drawing-Extraction Unit with Expert/Explorer Bridge"
