#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
واجهة الويب الموحدة لنظام بصيرة - Unified Web Interface
Basira System Unified Web Interface with AI-OOP Integration

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - AI-OOP Unified Edition
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# Import Unified Integration System
try:
    from integration.unified_system_integration import UnifiedSystemIntegration
    UNIFIED_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unified Integration not available: {e}")
    UNIFIED_INTEGRATION_AVAILABLE = False

# Import Revolutionary Foundation
try:
    from revolutionary_core.unified_revolutionary_foundation import get_revolutionary_foundation
    REVOLUTIONARY_FOUNDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Revolutionary Foundation not available: {e}")
    REVOLUTIONARY_FOUNDATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global integration system
integration_system = None

class UnifiedWebInterface:
    """واجهة الويب الموحدة مع تكامل AI-OOP"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        """تهيئة واجهة الويب الموحدة"""
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # تهيئة نظام التكامل
        self.integration_system = None
        self.system_status = "initializing"
        
        # إعداد المسارات
        self._setup_routes()
        
        # تهيئة النظام في خيط منفصل
        self._initialize_system_async()
    
    def _initialize_system_async(self):
        """تهيئة النظام بشكل غير متزامن"""
        def init_thread():
            try:
                if UNIFIED_INTEGRATION_AVAILABLE:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    self.integration_system = UnifiedSystemIntegration()
                    result = loop.run_until_complete(self.integration_system.initialize_system())
                    
                    if result.get("status") == "ready":
                        self.system_status = "ready"
                        logger.info("✅ نظام التكامل الموحد جاهز")
                    else:
                        self.system_status = "error"
                        logger.error("❌ فشل في تهيئة نظام التكامل")
                else:
                    self.system_status = "limited"
                    logger.warning("⚠️ نظام التكامل غير متوفر - وضع محدود")
            except Exception as e:
                self.system_status = "error"
                logger.error(f"❌ خطأ في تهيئة النظام: {e}")
        
        thread = threading.Thread(target=init_thread)
        thread.daemon = True
        thread.start()
    
    def _setup_routes(self):
        """إعداد مسارات الويب"""
        
        @self.app.route('/')
        def index():
            """الصفحة الرئيسية"""
            return render_template('unified_index.html')
        
        @self.app.route('/api/status')
        def get_status():
            """الحصول على حالة النظام"""
            if self.integration_system:
                status = self.integration_system.get_system_status()
                return jsonify({
                    "status": "success",
                    "system_status": self.system_status,
                    "detailed_status": status,
                    "ai_oop_enabled": REVOLUTIONARY_FOUNDATION_AVAILABLE,
                    "unified_integration": UNIFIED_INTEGRATION_AVAILABLE
                })
            else:
                return jsonify({
                    "status": "initializing",
                    "system_status": self.system_status,
                    "ai_oop_enabled": REVOLUTIONARY_FOUNDATION_AVAILABLE,
                    "unified_integration": UNIFIED_INTEGRATION_AVAILABLE
                })
        
        @self.app.route('/api/integration_report')
        def get_integration_report():
            """الحصول على تقرير التكامل"""
            if self.integration_system:
                report = self.integration_system.get_integration_report()
                return jsonify({
                    "status": "success",
                    "report": report
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Integration system not available"
                })
        
        @self.app.route('/api/revolutionary_learning', methods=['POST'])
        def revolutionary_learning():
            """اختبار التعلم الثوري"""
            try:
                data = request.get_json()
                situation = data.get('situation', {"complexity": 0.8, "novelty": 0.6})
                
                if self.integration_system and "learning" in self.integration_system.systems:
                    learning_system = self.integration_system.systems["learning"]["revolutionary_learning"]
                    
                    # اختبار قرار الخبير
                    expert_decision = learning_system.make_expert_decision(situation)
                    
                    # اختبار الاستكشاف
                    exploration_result = learning_system.explore_new_possibilities(situation)
                    
                    return jsonify({
                        "status": "success",
                        "expert_decision": expert_decision,
                        "exploration_result": exploration_result,
                        "ai_oop_applied": True
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Revolutionary learning system not available"
                    })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                })
        
        @self.app.route('/api/adaptive_equations', methods=['POST'])
        def adaptive_equations():
            """اختبار المعادلات المتكيفة"""
            try:
                data = request.get_json()
                pattern = data.get('pattern', [1, 2, 3, 4, 5])
                
                if self.integration_system and "learning" in self.integration_system.systems:
                    equation_system = self.integration_system.systems["learning"]["adaptive_equations"]
                    
                    # حل النمط
                    solution = equation_system.solve_pattern(pattern)
                    
                    return jsonify({
                        "status": "success",
                        "solution": solution,
                        "ai_oop_applied": True
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Adaptive equations system not available"
                    })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                })
        
        @self.app.route('/api/revolutionary_agent', methods=['POST'])
        def revolutionary_agent():
            """اختبار الوكيل الثوري"""
            try:
                data = request.get_json()
                situation = data.get('situation', {
                    "complexity": 0.8,
                    "urgency": 0.6,
                    "available_options": ["option_a", "option_b", "option_c"]
                })
                
                if self.integration_system and "learning" in self.integration_system.systems:
                    agent_system = self.integration_system.systems["learning"]["revolutionary_agent"]
                    
                    # اتخاذ قرار ثوري
                    decision = agent_system.make_revolutionary_decision(situation)
                    
                    return jsonify({
                        "status": "success",
                        "decision": {
                            "decision_id": decision.decision_id,
                            "decision_type": decision.decision_type,
                            "decision_value": decision.decision_value,
                            "confidence_level": decision.confidence_level,
                            "wisdom_basis": decision.wisdom_basis,
                            "expert_insight": decision.expert_insight,
                            "explorer_novelty": decision.explorer_novelty,
                            "basil_methodology_factor": decision.basil_methodology_factor,
                            "physics_resonance": decision.physics_resonance
                        },
                        "ai_oop_applied": True
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Revolutionary agent system not available"
                    })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                })
        
        @self.app.route('/api/mathematical_processing', methods=['POST'])
        def mathematical_processing():
            """معالجة رياضية"""
            try:
                data = request.get_json()
                equation = data.get('equation', 'x^2')
                
                if self.integration_system and "mathematical" in self.integration_system.systems:
                    gse = self.integration_system.systems["mathematical"]["general_shape_equation"]
                    
                    # إنشاء معادلة
                    result = gse.create_equation(equation, "mathematical")
                    
                    return jsonify({
                        "status": "success",
                        "result": result,
                        "mathematical_core_available": True
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Mathematical core not available"
                    })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                })
        
        @self.app.route('/api/test_ai_oop')
        def test_ai_oop():
            """اختبار AI-OOP"""
            try:
                if REVOLUTIONARY_FOUNDATION_AVAILABLE:
                    foundation = get_revolutionary_foundation()
                    
                    # اختبار إنشاء وحدات مختلفة
                    test_results = {}
                    
                    for unit_type in ["learning", "mathematical", "visual", "integration"]:
                        try:
                            from revolutionary_core.unified_revolutionary_foundation import create_revolutionary_unit
                            unit = create_revolutionary_unit(unit_type)
                            
                            test_input = {"test": True, "unit_type": unit_type}
                            output = unit.process_revolutionary_input(test_input)
                            
                            test_results[unit_type] = {
                                "success": True,
                                "unit_terms_count": len(unit.unit_terms),
                                "output": output
                            }
                        except Exception as e:
                            test_results[unit_type] = {
                                "success": False,
                                "error": str(e)
                            }
                    
                    return jsonify({
                        "status": "success",
                        "ai_oop_available": True,
                        "foundation_terms": len(foundation.revolutionary_terms),
                        "test_results": test_results
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Revolutionary Foundation not available",
                        "ai_oop_available": False
                    })
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "ai_oop_available": False
                })
    
    def run(self, debug=False):
        """تشغيل الخادم"""
        print("🌟" + "="*80 + "🌟")
        print("🌐 واجهة الويب الموحدة لنظام بصيرة")
        print("⚡ AI-OOP Integration + Unified Systems")
        print(f"🔗 الخادم يعمل على: http://{self.host}:{self.port}")
        print("🌟" + "="*80 + "🌟")
        
        self.app.run(host=self.host, port=self.port, debug=debug)


def create_unified_web_interface(host='0.0.0.0', port=5000):
    """إنشاء واجهة الويب الموحدة"""
    return UnifiedWebInterface(host, port)


if __name__ == "__main__":
    # إنشاء وتشغيل واجهة الويب الموحدة
    web_interface = create_unified_web_interface()
    web_interface.run(debug=True)
