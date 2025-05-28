#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تكامل نظام تفسير الأحلام مع نظام بصيرة الرئيسي

هذا الملف يدمج نظام تفسير الأحلام المتقدم وفق نظرية باسل
مع باقي مكونات نظام بصيرة لتوفير تجربة متكاملة.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import json
from datetime import datetime

# استيراد مكونات بصيرة
from ..core.central_thinking.thinking_core import CentralThinkingCore
from ..symbolic_processing.letter_semantics.semantic_analyzer import LetterSemanticAnalyzer
from ..language_generation.creative_text.text_generator import CreativeTextGenerator
from ..visual_processing.image_generation.image_generator import ImageGenerator
from ..learning_systems.reinforcement_learning.rl_system import ReinforcementLearningSystem

# استيراد نظام تفسير الأحلام
from .basil_dream_system import (
    BasilDreamInterpreter,
    DreamerProfile,
    BasilDreamInterpretation,
    DreamType,
    create_basil_dream_interpreter,
    create_dreamer_profile
)

class BasiraDreamSystem:
    """
    نظام تفسير الأحلام المتكامل في بصيرة
    يدمج تفسير الأحلام مع قدرات النظام الأخرى
    """
    
    def __init__(self, thinking_core: CentralThinkingCore = None):
        self.thinking_core = thinking_core or CentralThinkingCore()
        self.dream_interpreter = create_basil_dream_interpreter()
        self.user_profiles = {}  # ملفات المستخدمين
        self.interpretation_sessions = {}  # جلسات التفسير
        self.logger = logging.getLogger("basira_dream_system")
        
        # تسجيل نظام الأحلام في نواة التفكير
        self.thinking_core.register_module("dream_interpreter", self.dream_interpreter)
        
        # تسجيل استراتيجية تفكير خاصة بالأحلام
        self.thinking_core.register_thinking_strategy("dream_analysis", self._dream_analysis_strategy)
        
        self.logger.info("تم تهيئة نظام تفسير الأحلام المتكامل")
    
    def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> DreamerProfile:
        """
        إنشاء ملف شخصي للمستخدم
        
        Args:
            user_id: معرف المستخدم
            profile_data: بيانات الملف الشخصي
        
        Returns:
            ملف شخصي للرائي
        """
        profile = create_dreamer_profile(**profile_data)
        self.user_profiles[user_id] = profile
        
        self.logger.info(f"تم إنشاء ملف شخصي للمستخدم: {user_id}")
        return profile
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[DreamerProfile]:
        """
        تحديث ملف شخصي للمستخدم
        
        Args:
            user_id: معرف المستخدم
            updates: التحديثات المطلوبة
        
        Returns:
            الملف الشخصي المحدث أو None إذا لم يوجد
        """
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        
        # تحديث الحقول
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        self.logger.info(f"تم تحديث ملف المستخدم: {user_id}")
        return profile
    
    def interpret_dream_comprehensive(self, user_id: str, dream_text: str, 
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        تفسير شامل للحلم مع استخدام قدرات بصيرة المتكاملة
        
        Args:
            user_id: معرف المستخدم
            dream_text: نص الحلم
            context: سياق إضافي
        
        Returns:
            تفسير شامل مع تحليلات إضافية
        """
        # التحقق من وجود ملف المستخدم
        if user_id not in self.user_profiles:
            return {
                "success": False,
                "error": "لم يتم العثور على ملف المستخدم. يرجى إنشاء ملف شخصي أولاً."
            }
        
        dreamer_profile = self.user_profiles[user_id]
        context = context or {}
        
        # تفسير الحلم الأساسي
        basic_interpretation = self.dream_interpreter.interpret_dream(
            dream_text, dreamer_profile, context
        )
        
        # تحليل إضافي باستخدام نواة التفكير
        thinking_result = self.thinking_core.think({
            "dream_text": dream_text,
            "basic_interpretation": basic_interpretation.to_dict(),
            "dreamer_profile": dreamer_profile.__dict__,
            "context": context
        }, strategy="dream_analysis")
        
        # توليد تحليل بصري للحلم (إذا كان متاحاً)
        visual_analysis = self._generate_visual_analysis(dream_text, basic_interpretation)
        
        # توليد نص إبداعي يشرح الحلم
        narrative_explanation = self._generate_narrative_explanation(
            dream_text, basic_interpretation, dreamer_profile
        )
        
        # إنشاء جلسة تفسير
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "dream_text": dream_text,
            "basic_interpretation": basic_interpretation.to_dict(),
            "thinking_analysis": thinking_result,
            "visual_analysis": visual_analysis,
            "narrative_explanation": narrative_explanation,
            "timestamp": datetime.now().isoformat(),
            "feedback": None
        }
        
        self.interpretation_sessions[session_id] = session_data
        
        # إنشاء النتيجة الشاملة
        comprehensive_result = {
            "success": True,
            "session_id": session_id,
            "basic_interpretation": basic_interpretation.to_dict(),
            "advanced_analysis": {
                "thinking_insights": thinking_result,
                "visual_elements": visual_analysis,
                "narrative_story": narrative_explanation
            },
            "recommendations": self._generate_enhanced_recommendations(
                basic_interpretation, thinking_result, dreamer_profile
            ),
            "follow_up_questions": self._generate_follow_up_questions(
                basic_interpretation, dreamer_profile
            )
        }
        
        self.logger.info(f"تم إنجاز تفسير شامل للمستخدم {user_id} في الجلسة {session_id}")
        
        return comprehensive_result
    
    def _dream_analysis_strategy(self, thinking_core, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        استراتيجية تفكير خاصة بتحليل الأحلام
        """
        dream_text = input_data.get("dream_text", "")
        basic_interpretation = input_data.get("basic_interpretation", {})
        dreamer_profile = input_data.get("dreamer_profile", {})
        
        analysis = {
            "psychological_insights": self._analyze_psychological_patterns(
                dream_text, basic_interpretation, dreamer_profile
            ),
            "symbolic_connections": self._find_symbolic_connections(
                basic_interpretation.get("elements", [])
            ),
            "life_guidance": self._extract_life_guidance(
                basic_interpretation, dreamer_profile
            ),
            "spiritual_dimensions": self._explore_spiritual_dimensions(
                dream_text, basic_interpretation, dreamer_profile
            )
        }
        
        return analysis
    
    def _analyze_psychological_patterns(self, dream_text: str, 
                                      interpretation: Dict[str, Any], 
                                      profile: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل الأنماط النفسية في الحلم"""
        patterns = {
            "emotional_state": "متوازن",
            "stress_indicators": [],
            "growth_opportunities": [],
            "subconscious_messages": []
        }
        
        # تحليل الحالة العاطفية
        elements = interpretation.get("elements", [])
        positive_count = sum(1 for elem in elements 
                           if any(meaning in ["فرح", "سعادة", "نجاح", "خير"] 
                                 for meaning in elem.get("symbolic_meanings", [])))
        
        negative_count = sum(1 for elem in elements 
                           if any(meaning in ["خوف", "قلق", "حزن", "فشل"] 
                                 for meaning in elem.get("symbolic_meanings", [])))
        
        if positive_count > negative_count:
            patterns["emotional_state"] = "إيجابي"
        elif negative_count > positive_count:
            patterns["emotional_state"] = "يحتاج انتباه"
        
        # تحليل مؤشرات التوتر
        stress_words = ["خوف", "قلق", "هروب", "مطاردة", "سقوط"]
        for word in stress_words:
            if word in dream_text:
                patterns["stress_indicators"].append(f"وجود كلمة '{word}' قد تشير لتوتر")
        
        # فرص النمو
        growth_words = ["طيران", "صعود", "نور", "ماء", "شجرة"]
        for word in growth_words:
            if word in dream_text:
                patterns["growth_opportunities"].append(f"'{word}' يشير لفرصة نمو")
        
        return patterns
    
    def _find_symbolic_connections(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """العثور على الروابط الرمزية بين عناصر الحلم"""
        connections = []
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                # البحث عن معاني مشتركة
                meanings1 = set(elem1.get("symbolic_meanings", []))
                meanings2 = set(elem2.get("symbolic_meanings", []))
                common_meanings = meanings1.intersection(meanings2)
                
                if common_meanings:
                    connections.append({
                        "element1": elem1.get("element", ""),
                        "element2": elem2.get("element", ""),
                        "connection_type": "معاني مشتركة",
                        "shared_meanings": list(common_meanings)
                    })
        
        return connections
    
    def _extract_life_guidance(self, interpretation: Dict[str, Any], 
                             profile: Dict[str, Any]) -> List[str]:
        """استخراج إرشادات حياتية من التفسير"""
        guidance = []
        
        # إرشادات بناءً على نوع الحلم
        dream_type = interpretation.get("dream_type", "")
        
        if dream_type == "رؤيا_صادقة":
            guidance.append("هذه رؤيا صادقة، تأمل في رسالتها واستعن بالله في تطبيقها")
        
        # إرشادات بناءً على المهنة
        profession = profile.get("profession", "")
        if profession:
            guidance.append(f"كونك {profession}، قد يكون للحلم علاقة بمجال عملك")
        
        # إرشادات بناءً على الاهتمامات
        interests = profile.get("interests", [])
        if interests:
            guidance.append(f"اهتماماتك بـ {', '.join(interests)} قد تكون مفتاحاً لفهم الحلم")
        
        return guidance
    
    def _explore_spiritual_dimensions(self, dream_text: str, 
                                    interpretation: Dict[str, Any], 
                                    profile: Dict[str, Any]) -> Dict[str, Any]:
        """استكشاف الأبعاد الروحية للحلم"""
        spiritual = {
            "religious_symbols": [],
            "spiritual_messages": [],
            "recommended_practices": []
        }
        
        # البحث عن رموز دينية
        religious_words = ["مسجد", "قرآن", "نبي", "ملك", "جنة", "نار", "صلاة", "حج"]
        for word in religious_words:
            if word in dream_text:
                spiritual["religious_symbols"].append(word)
        
        # رسائل روحية
        if spiritual["religious_symbols"]:
            spiritual["spiritual_messages"].append("الحلم يحمل رسالة روحية قوية")
        
        # ممارسات موصى بها
        if profile.get("religion") == "إسلام":
            spiritual["recommended_practices"].extend([
                "الإكثار من الاستغفار",
                "قراءة القرآن بتدبر",
                "الدعاء والتضرع إلى الله"
            ])
        
        return spiritual
    
    def _generate_visual_analysis(self, dream_text: str, 
                                interpretation: BasilDreamInterpretation) -> Dict[str, Any]:
        """توليد تحليل بصري للحلم"""
        visual_analysis = {
            "dominant_colors": [],
            "visual_themes": [],
            "spatial_elements": [],
            "movement_patterns": []
        }
        
        # تحليل الألوان المذكورة
        colors = ["أبيض", "أسود", "أحمر", "أخضر", "أزرق", "أصفر"]
        for color in colors:
            if color in dream_text:
                visual_analysis["dominant_colors"].append(color)
        
        # تحليل المواضيع البصرية
        if any(elem.element in ["شمس", "نور", "ضوء"] for elem in interpretation.elements):
            visual_analysis["visual_themes"].append("إضاءة وإشراق")
        
        if any(elem.element in ["ماء", "بحر", "نهر"] for elem in interpretation.elements):
            visual_analysis["visual_themes"].append("عناصر مائية")
        
        # تحليل العناصر المكانية
        spatial_words = ["فوق", "تحت", "يمين", "شمال", "أمام", "خلف"]
        for word in spatial_words:
            if word in dream_text:
                visual_analysis["spatial_elements"].append(word)
        
        # تحليل أنماط الحركة
        movement_words = ["طيران", "سقوط", "جري", "مشي", "صعود", "نزول"]
        for word in movement_words:
            if word in dream_text:
                visual_analysis["movement_patterns"].append(word)
        
        return visual_analysis
    
    def _generate_narrative_explanation(self, dream_text: str, 
                                      interpretation: BasilDreamInterpretation,
                                      dreamer_profile: DreamerProfile) -> str:
        """توليد شرح سردي للحلم"""
        narrative = f"قصة حلم {dreamer_profile.name}:\n\n"
        
        narrative += f"في ليلة هادئة، رأى {dreamer_profile.name} حلماً يحمل رسائل عميقة. "
        
        # إضافة وصف العناصر
        if interpretation.elements:
            narrative += "ظهرت في الحلم عناصر مهمة: "
            for elem in interpretation.elements[:3]:
                narrative += f"{elem.element} الذي يرمز إلى {', '.join(elem.symbolic_meanings[:2])}، "
        
        # إضافة التفسير الشامل
        narrative += f"\n\nهذا الحلم من نوع {interpretation.dream_type.value}، "
        narrative += f"ويحمل رسالة {interpretation.overall_message[:100]}..."
        
        # إضافة نصيحة شخصية
        if dreamer_profile.profession:
            narrative += f"\n\nبصفتك {dreamer_profile.profession}، قد تجد في هذا الحلم إرشاداً لمسيرتك المهنية."
        
        return narrative
    
    def _generate_enhanced_recommendations(self, interpretation: BasilDreamInterpretation,
                                         thinking_result: Dict[str, Any],
                                         dreamer_profile: DreamerProfile) -> List[str]:
        """توليد توصيات محسنة"""
        recommendations = list(interpretation.recommendations)
        
        # إضافة توصيات من التحليل المتقدم
        if thinking_result.get("psychological_insights", {}).get("stress_indicators"):
            recommendations.append("انتبه لمستوى التوتر في حياتك وحاول تقليله")
        
        if thinking_result.get("spiritual_dimensions", {}).get("religious_symbols"):
            recommendations.append("اهتم بالجانب الروحي في حياتك")
        
        # توصيات شخصية
        if dreamer_profile.current_concerns:
            recommendations.append("ربط الحلم بهمومك الحالية قد يوضح الرسالة")
        
        return recommendations
    
    def _generate_follow_up_questions(self, interpretation: BasilDreamInterpretation,
                                    dreamer_profile: DreamerProfile) -> List[str]:
        """توليد أسئلة متابعة"""
        questions = [
            "هل تذكر تفاصيل أخرى عن الحلم؟",
            "ما هو شعورك العام تجاه هذا الحلم؟",
            "هل رأيت أحلاماً مشابهة من قبل؟"
        ]
        
        # أسئلة بناءً على العناصر
        if interpretation.elements:
            element = interpretation.elements[0].element
            questions.append(f"ما هو انطباعك الشخصي عن {element} في الحلم؟")
        
        # أسئلة بناءً على نوع الحلم
        if interpretation.dream_type == DreamType.TRUE_VISION:
            questions.append("هل تشعر أن هذا الحلم يحمل رسالة مهمة لك؟")
        
        return questions
    
    def record_user_feedback(self, session_id: str, feedback: Dict[str, Any]) -> bool:
        """تسجيل تقييم المستخدم للتفسير"""
        if session_id not in self.interpretation_sessions:
            return False
        
        self.interpretation_sessions[session_id]["feedback"] = feedback
        
        # تحديث نظام التعلم المعزز
        if "rating" in feedback:
            # يمكن إضافة منطق التعلم هنا
            pass
        
        self.logger.info(f"تم تسجيل تقييم للجلسة: {session_id}")
        return True
    
    def get_user_dream_history(self, user_id: str) -> List[Dict[str, Any]]:
        """الحصول على تاريخ أحلام المستخدم"""
        user_sessions = [
            session for session in self.interpretation_sessions.values()
            if session["user_id"] == user_id
        ]
        
        return sorted(user_sessions, key=lambda x: x["timestamp"], reverse=True)
    
    def export_interpretation_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """تصدير تقرير شامل للتفسير"""
        if session_id not in self.interpretation_sessions:
            return None
        
        session = self.interpretation_sessions[session_id]
        
        report = {
            "report_title": f"تقرير تفسير الحلم - {session_id}",
            "generated_at": datetime.now().isoformat(),
            "session_data": session,
            "summary": {
                "dream_type": session["basic_interpretation"]["dream_type"],
                "confidence_level": session["basic_interpretation"]["confidence_level"],
                "main_elements": [elem["element"] for elem in session["basic_interpretation"]["elements"][:5]],
                "key_message": session["basic_interpretation"]["overall_message"][:200] + "..."
            }
        }
        
        return report

# دالة مساعدة لإنشاء النظام المتكامل
def create_basira_dream_system(thinking_core: CentralThinkingCore = None) -> BasiraDreamSystem:
    """إنشاء نظام تفسير الأحلام المتكامل"""
    return BasiraDreamSystem(thinking_core)
