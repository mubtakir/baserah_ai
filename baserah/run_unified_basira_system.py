#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مشغل نظام بصيرة الموحد - Unified Basira System Launcher
المشغل الرئيسي لجميع مكونات نظام بصيرة الثوري الموحد

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - Unified System Launcher
"""

import os
import sys
import asyncio
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add baserah_system to path
sys.path.insert(0, 'baserah_system')

print("🌟" + "="*120 + "🌟")
print("🚀 مشغل نظام بصيرة الموحد - Unified Basira System Launcher")
print("⚡ النظام الثوري الخبير/المستكشف مع AI-OOP الكامل")
print("🧠 تكامل شامل لجميع مكونات النظام الثوري")
print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل 🌟")
print("🌟" + "="*120 + "🌟")

class UnifiedBasiraSystemLauncher:
    """مشغل نظام بصيرة الموحد"""
    
    def __init__(self):
        """تهيئة المشغل"""
        self.integration_system = None
        self.web_interface = None
        self.desktop_interface = None
        self.system_status = "initializing"
        
    async def initialize_system(self) -> Dict[str, Any]:
        """تهيئة النظام الموحد"""
        print("\n🔧 تهيئة النظام الموحد...")
        
        try:
            # 1. تهيئة الأساس الثوري
            foundation_result = await self._initialize_revolutionary_foundation()
            
            # 2. تهيئة نظام التكامل
            integration_result = await self._initialize_integration_system()
            
            # 3. تهيئة الأنظمة الثورية
            systems_result = await self._initialize_revolutionary_systems()
            
            # 4. تهيئة الواجهات
            interfaces_result = await self._initialize_interfaces()
            
            initialization_report = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "revolutionary_foundation": foundation_result,
                    "integration_system": integration_result,
                    "revolutionary_systems": systems_result,
                    "interfaces": interfaces_result
                }
            }
            
            self.system_status = "ready"
            print("✅ تم تهيئة النظام الموحد بنجاح!")
            
            return initialization_report
            
        except Exception as e:
            self.system_status = "error"
            print(f"❌ خطأ في تهيئة النظام: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _initialize_revolutionary_foundation(self) -> Dict[str, Any]:
        """تهيئة الأساس الثوري"""
        print("🏗️ تهيئة الأساس الثوري...")
        
        try:
            from revolutionary_core.unified_revolutionary_foundation import (
                get_revolutionary_foundation,
                create_revolutionary_unit
            )
            
            foundation = get_revolutionary_foundation()
            
            # اختبار إنشاء وحدات أساسية
            test_units = []
            for unit_type in ["learning", "mathematical", "visual", "integration"]:
                try:
                    unit = create_revolutionary_unit(unit_type)
                    test_units.append(unit_type)
                except Exception as e:
                    print(f"⚠️ تحذير: فشل في إنشاء وحدة {unit_type}: {e}")
            
            return {
                "status": "ready",
                "foundation_terms": len(foundation.revolutionary_terms),
                "test_units_created": test_units,
                "ai_oop_applied": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _initialize_integration_system(self) -> Dict[str, Any]:
        """تهيئة نظام التكامل"""
        print("🔗 تهيئة نظام التكامل...")
        
        try:
            from integration.unified_system_integration import UnifiedSystemIntegration
            
            self.integration_system = UnifiedSystemIntegration()
            init_result = await self.integration_system.initialize_system()
            
            return {
                "status": init_result.get("status", "unknown"),
                "connected_systems": init_result.get("connected_systems", 0),
                "ai_oop_applied": init_result.get("ai_oop_applied", False)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _initialize_revolutionary_systems(self) -> Dict[str, Any]:
        """تهيئة الأنظمة الثورية"""
        print("🧠 تهيئة الأنظمة الثورية...")
        
        systems_status = {}
        
        # تهيئة التعلم الثوري
        try:
            from learning.reinforcement.innovative_rl_unified import create_unified_revolutionary_learning_system
            learning_system = create_unified_revolutionary_learning_system()
            systems_status["revolutionary_learning"] = {"status": "ready", "type": "learning"}
        except Exception as e:
            systems_status["revolutionary_learning"] = {"status": "error", "error": str(e)}
        
        # تهيئة المعادلات المتكيفة
        try:
            from learning.reinforcement.equation_based_rl_unified import create_unified_adaptive_equation_system
            equation_system = create_unified_adaptive_equation_system()
            systems_status["adaptive_equations"] = {"status": "ready", "type": "mathematical"}
        except Exception as e:
            systems_status["adaptive_equations"] = {"status": "error", "error": str(e)}
        
        # تهيئة الوكيل الثوري
        try:
            from learning.innovative_reinforcement.agent_unified import create_unified_revolutionary_agent
            agent_system = create_unified_revolutionary_agent()
            systems_status["revolutionary_agent"] = {"status": "ready", "type": "agent"}
        except Exception as e:
            systems_status["revolutionary_agent"] = {"status": "error", "error": str(e)}
        
        # تهيئة تفسير الأحلام الثوري
        try:
            from dream_interpretation.revolutionary_dream_interpreter_unified import create_unified_revolutionary_dream_interpreter
            dream_interpreter = create_unified_revolutionary_dream_interpreter()
            systems_status["dream_interpretation"] = {"status": "ready", "type": "interpretation"}
        except Exception as e:
            systems_status["dream_interpretation"] = {"status": "error", "error": str(e)}
        
        ready_systems = sum(1 for system in systems_status.values() if system.get("status") == "ready")
        total_systems = len(systems_status)
        
        return {
            "systems_status": systems_status,
            "ready_systems": ready_systems,
            "total_systems": total_systems,
            "success_rate": (ready_systems / total_systems) * 100 if total_systems > 0 else 0
        }
    
    async def _initialize_interfaces(self) -> Dict[str, Any]:
        """تهيئة الواجهات"""
        print("🌐 تهيئة الواجهات...")
        
        interfaces_status = {}
        
        # تهيئة واجهة الويب
        try:
            from interfaces.web.unified_web_interface import create_unified_web_interface
            self.web_interface = create_unified_web_interface(host='127.0.0.1', port=5000)
            interfaces_status["web_interface"] = {
                "status": "ready",
                "host": "127.0.0.1",
                "port": 5000,
                "url": "http://127.0.0.1:5000"
            }
        except Exception as e:
            interfaces_status["web_interface"] = {"status": "error", "error": str(e)}
        
        # تهيئة واجهة سطح المكتب
        try:
            from interfaces.desktop.unified_desktop_interface import create_unified_desktop_interface
            self.desktop_interface = create_unified_desktop_interface()
            interfaces_status["desktop_interface"] = {
                "status": "ready",
                "type": "PyQt5/Tkinter/Console"
            }
        except Exception as e:
            interfaces_status["desktop_interface"] = {"status": "error", "error": str(e)}
        
        return interfaces_status
    
    async def launch_web_interface(self, host: str = '127.0.0.1', port: int = 5000):
        """تشغيل واجهة الويب"""
        print(f"\n🌐 تشغيل واجهة الويب على {host}:{port}...")
        
        try:
            if self.web_interface:
                await self.web_interface.run_async(host=host, port=port)
            else:
                print("❌ واجهة الويب غير متوفرة")
        except Exception as e:
            print(f"❌ خطأ في تشغيل واجهة الويب: {e}")
    
    def launch_desktop_interface(self):
        """تشغيل واجهة سطح المكتب"""
        print("\n🖥️ تشغيل واجهة سطح المكتب...")
        
        try:
            if self.desktop_interface:
                return self.desktop_interface.run()
            else:
                print("❌ واجهة سطح المكتب غير متوفرة")
                return None
        except Exception as e:
            print(f"❌ خطأ في تشغيل واجهة سطح المكتب: {e}")
            return None
    
    async def run_comprehensive_test(self):
        """تشغيل اختبار شامل"""
        print("\n🔬 تشغيل اختبار شامل...")
        
        try:
            # استيراد وتشغيل الاختبار الشامل
            from test_complete_unified_system import ComprehensiveSystemTester
            
            tester = ComprehensiveSystemTester()
            test_result = await tester.run_complete_test()
            
            return test_result
            
        except Exception as e:
            print(f"❌ خطأ في الاختبار الشامل: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """الحصول على حالة النظام"""
        if self.integration_system:
            return self.integration_system.get_system_status()
        else:
            return {
                "overall_status": self.system_status,
                "integration_available": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_report(self) -> Dict[str, Any]:
        """الحصول على تقرير النظام"""
        if self.integration_system:
            return self.integration_system.get_integration_report()
        else:
            return {
                "status": "limited",
                "message": "نظام التكامل غير متوفر",
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description="مشغل نظام بصيرة الموحد")
    parser.add_argument("--mode", choices=["web", "desktop", "test", "status"], 
                       default="desktop", help="وضع التشغيل")
    parser.add_argument("--host", default="127.0.0.1", help="عنوان الخادم للواجهة الويب")
    parser.add_argument("--port", type=int, default=5000, help="منفذ الخادم للواجهة الويب")
    parser.add_argument("--no-init", action="store_true", help="تخطي تهيئة النظام")
    
    args = parser.parse_args()
    
    # إنشاء المشغل
    launcher = UnifiedBasiraSystemLauncher()
    
    # تهيئة النظام (إلا إذا تم تخطيها)
    if not args.no_init:
        init_result = await launcher.initialize_system()
        
        if init_result.get("status") != "success":
            print("❌ فشل في تهيئة النظام. التشغيل في وضع محدود.")
    
    # تشغيل الوضع المطلوب
    if args.mode == "web":
        print(f"\n🌐 تشغيل واجهة الويب على http://{args.host}:{args.port}")
        print("للوصول للنظام، افتح المتصفح وانتقل إلى العنوان أعلاه")
        print("اضغط Ctrl+C للإيقاف")
        
        try:
            await launcher.launch_web_interface(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\n🛑 تم إيقاف واجهة الويب")
    
    elif args.mode == "desktop":
        print("\n🖥️ تشغيل واجهة سطح المكتب...")
        result = launcher.launch_desktop_interface()
        if result is not None:
            print("✅ تم إغلاق واجهة سطح المكتب بنجاح")
        else:
            print("❌ فشل في تشغيل واجهة سطح المكتب")
    
    elif args.mode == "test":
        print("\n🔬 تشغيل الاختبار الشامل...")
        test_result = await launcher.run_comprehensive_test()
        
        if test_result:
            verdict = test_result.get("final_verdict", {})
            print(f"\n🎯 نتيجة الاختبار: {verdict.get('status', 'UNKNOWN')}")
            print(f"📝 الرسالة: {verdict.get('message', 'لا توجد رسالة')}")
            
            # حفظ النتيجة
            with open("system_test_result.json", "w", encoding="utf-8") as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2)
            print("💾 تم حفظ نتيجة الاختبار في: system_test_result.json")
        else:
            print("❌ فشل في تشغيل الاختبار الشامل")
    
    elif args.mode == "status":
        print("\n📊 حالة النظام:")
        print("-" * 50)
        
        status = launcher.get_system_status()
        print(f"الحالة العامة: {status.get('overall_status', 'غير معروف')}")
        print(f"AI-OOP مطبق: {status.get('ai_oop_applied', False)}")
        print(f"الأنظمة المتصلة: {status.get('connected_systems', 0)}")
        
        report = launcher.get_system_report()
        if "integration_summary" in report:
            summary = report["integration_summary"]
            print(f"معدل النجاح: {summary.get('success_rate', 0):.1f}%")
            print(f"المكونات الجاهزة: {summary.get('ready_components', 0)}")
            print(f"إجمالي المكونات: {summary.get('total_components', 0)}")
    
    print("\n🌟 شكراً لاستخدام نظام بصيرة الموحد! 🌟")
    print("🎯 إبداع باسل يحيى عبدالله من العراق/الموصل محفوظ ومطور!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف النظام بواسطة المستخدم")
    except Exception as e:
        print(f"\n❌ خطأ في تشغيل النظام: {e}")
        sys.exit(1)
