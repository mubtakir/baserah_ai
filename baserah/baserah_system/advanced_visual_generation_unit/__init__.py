#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visual Generation Unit Package
حزمة وحدة التوليد البصري المتقدمة

This package contains advanced visual generation capabilities including:
- Revolutionary image and video generation
- Advanced artistic drawing engine with exceptional quality
- Comprehensive visual system with physics integration
- Expert analysis and quality optimization

هذه الحزمة تحتوي على قدرات التوليد البصري المتقدمة تشمل:
- توليد ثوري للصور والفيديو
- محرك رسم فني متقدم بجودة استثنائية
- نظام بصري شامل مع تكامل فيزيائي
- تحليل خبير وتحسين الجودة

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

from .revolutionary_image_video_generator import (
    RevolutionaryImageVideoGenerator,
    VisualGenerationRequest,
    VisualGenerationResult
)

from .advanced_artistic_drawing_engine import (
    AdvancedArtisticDrawingEngine,
    ArtisticDrawingRequest,
    ArtisticDrawingResult
)

from .comprehensive_visual_system import (
    ComprehensiveVisualSystem,
    ComprehensiveVisualRequest,
    ComprehensiveVisualResult
)

__all__ = [
    # Revolutionary Image/Video Generator
    'RevolutionaryImageVideoGenerator',
    'VisualGenerationRequest',
    'VisualGenerationResult',
    
    # Advanced Artistic Drawing Engine
    'AdvancedArtisticDrawingEngine',
    'ArtisticDrawingRequest',
    'ArtisticDrawingResult',
    
    # Comprehensive Visual System
    'ComprehensiveVisualSystem',
    'ComprehensiveVisualRequest',
    'ComprehensiveVisualResult'
]

__version__ = "1.0.0"
__author__ = "Basil Yahya Abdullah - Iraq/Mosul"
__description__ = "Advanced Visual Generation Unit with Revolutionary Capabilities"

# Package information
PACKAGE_INFO = {
    "name": "Advanced Visual Generation Unit",
    "arabic_name": "وحدة التوليد البصري المتقدمة",
    "version": __version__,
    "author": __author__,
    "capabilities": [
        "Revolutionary image generation",
        "Advanced video creation", 
        "Exceptional artistic drawing",
        "Physics-integrated rendering",
        "Expert quality analysis",
        "Comprehensive visual workflows"
    ],
    "arabic_capabilities": [
        "توليد صور ثوري",
        "إنشاء فيديو متقدم",
        "رسم فني استثنائي", 
        "رسم متكامل مع الفيزياء",
        "تحليل جودة خبير",
        "سير عمل بصري شامل"
    ]
}

def get_package_info():
    """Get package information"""
    return PACKAGE_INFO

def print_package_info():
    """Print package information in Arabic and English"""
    print("🌟" + "="*70 + "🌟")
    print(f"📦 {PACKAGE_INFO['name']}")
    print(f"📦 {PACKAGE_INFO['arabic_name']}")
    print(f"👨‍💻 المطور: {PACKAGE_INFO['author']}")
    print(f"📦 الإصدار: {PACKAGE_INFO['version']}")
    print("🌟" + "="*70 + "🌟")
    
    print("\n🎯 القدرات المتاحة:")
    for i, (eng, ar) in enumerate(zip(PACKAGE_INFO['capabilities'], PACKAGE_INFO['arabic_capabilities']), 1):
        print(f"   {i}. {ar} ({eng})")
    
    print("\n✨ مكونات الحزمة:")
    print("   🖼️ مولد الصور والفيديو الثوري")
    print("   🎨 محرك الرسم الفني المتقدم")
    print("   🌟 النظام البصري الشامل")
    print("🌟" + "="*70 + "🌟")

# Auto-print package info when imported
if __name__ != "__main__":
    print("✅ تم تحميل وحدة التوليد البصري المتقدمة")
