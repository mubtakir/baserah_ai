#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Generation Web Interface for Basira System
واجهة الويب للتوليد البصري - نظام بصيرة

Interactive web interface for the comprehensive visual generation system
with real-time preview, advanced controls, and expert analysis.

واجهة ويب تفاعلية للنظام البصري الشامل مع معاينة فورية
وتحكم متقدم وتحليل خبير.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import sys
import os
import json
from datetime import datetime
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revolutionary_database import ShapeEntity, RevolutionaryShapeDatabase
from comprehensive_visual_system import ComprehensiveVisualSystem, ComprehensiveVisualRequest

app = Flask(__name__)
app.secret_key = 'basira_visual_generation_2024'

# تهيئة النظام البصري الشامل
visual_system = ComprehensiveVisualSystem()
shape_db = RevolutionaryShapeDatabase()

# إحصائيات الواجهة
interface_stats = {
    "total_requests": 0,
    "successful_generations": 0,
    "active_users": 0,
    "start_time": datetime.now()
}

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/api/shapes')
def get_shapes():
    """الحصول على قائمة الأشكال المتاحة"""
    try:
        shapes = shape_db.get_all_shapes()
        shapes_data = []
        
        for shape in shapes:
            shapes_data.append({
                "id": shape.id,
                "name": shape.name,
                "category": shape.category,
                "color": shape.color_properties.get("dominant_color", [100, 150, 200]),
                "description": f"{shape.name} من فئة {shape.category}"
            })
        
        return jsonify({
            "success": True,
            "shapes": shapes_data,
            "total": len(shapes_data)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"خطأ في جلب الأشكال: {e}"
        })

@app.route('/api/generate', methods=['POST'])
def generate_visual_content():
    """توليد المحتوى البصري"""
    try:
        data = request.get_json()
        
        # تحديث الإحصائيات
        interface_stats["total_requests"] += 1
        
        # الحصول على الشكل المحدد
        shape_id = data.get('shape_id')
        shape = shape_db.get_shape_by_id(shape_id)
        
        if not shape:
            return jsonify({
                "success": False,
                "error": "الشكل المحدد غير موجود"
            })
        
        # إنشاء طلب التوليد
        request_data = ComprehensiveVisualRequest(
            shape=shape,
            output_types=data.get('output_types', ['image']),
            quality_level=data.get('quality_level', 'high'),
            artistic_styles=data.get('artistic_styles', ['photorealistic']),
            physics_simulation=data.get('physics_simulation', True),
            expert_analysis=data.get('expert_analysis', True),
            custom_effects=data.get('custom_effects', []),
            output_resolution=tuple(data.get('output_resolution', [1920, 1080])),
            animation_duration=data.get('animation_duration', 5.0)
        )
        
        # تنفيذ التوليد في خيط منفصل
        generation_thread = threading.Thread(
            target=process_generation_request,
            args=(request_data, interface_stats["total_requests"])
        )
        generation_thread.start()
        
        return jsonify({
            "success": True,
            "message": "تم بدء عملية التوليد",
            "request_id": interface_stats["total_requests"],
            "estimated_time": "5-10 ثواني"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"خطأ في طلب التوليد: {e}"
        })

def process_generation_request(request_data, request_id):
    """معالجة طلب التوليد"""
    try:
        # تنفيذ التوليد
        result = visual_system.create_comprehensive_visual_content(request_data)
        
        # حفظ النتيجة
        result_file = f"generation_result_{request_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "success": result.success,
                "generated_content": result.generated_content,
                "quality_metrics": result.quality_metrics,
                "artistic_scores": result.artistic_scores,
                "expert_analysis": result.expert_analysis,
                "physics_compliance": result.physics_compliance,
                "recommendations": result.recommendations,
                "processing_time": result.total_processing_time,
                "metadata": result.metadata
            }, f, ensure_ascii=False, indent=2)
        
        if result.success:
            interface_stats["successful_generations"] += 1
            
    except Exception as e:
        print(f"خطأ في معالجة طلب التوليد {request_id}: {e}")

@app.route('/api/result/<int:request_id>')
def get_generation_result(request_id):
    """الحصول على نتيجة التوليد"""
    try:
        result_file = f"generation_result_{request_id}.json"
        
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            return jsonify({
                "success": True,
                "result": result_data,
                "status": "completed"
            })
        else:
            return jsonify({
                "success": False,
                "status": "processing",
                "message": "التوليد قيد المعالجة..."
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"خطأ في جلب النتيجة: {e}"
        })

@app.route('/api/presets')
def get_presets():
    """الحصول على الإعدادات المسبقة"""
    return jsonify({
        "success": True,
        "presets": {
            "quality_levels": {
                "standard": "جودة عادية (1280x720)",
                "high": "جودة عالية (1920x1080)", 
                "ultra": "جودة فائقة (2560x1440)",
                "masterpiece": "تحفة فنية (3840x2160)"
            },
            "artistic_styles": {
                "photorealistic": "واقعي فوتوغرافي",
                "impressionist": "انطباعي",
                "watercolor": "ألوان مائية",
                "oil_painting": "رسم زيتي",
                "digital_art": "فن رقمي",
                "anime": "أنمي",
                "abstract": "تجريدي",
                "sketch": "رسم تخطيطي"
            },
            "output_types": {
                "image": "صورة ثابتة",
                "video": "فيديو متحرك",
                "artwork": "عمل فني",
                "animation": "رسم متحرك"
            },
            "visual_effects": {
                "glow": "توهج",
                "blur": "ضبابية", 
                "sharpen": "حدة",
                "emboss": "نقش",
                "vintage": "كلاسيكي",
                "neon": "نيون"
            }
        }
    })

@app.route('/api/stats')
def get_stats():
    """إحصائيات النظام"""
    system_stats = visual_system.get_system_statistics()
    
    uptime = (datetime.now() - interface_stats["start_time"]).total_seconds()
    
    return jsonify({
        "success": True,
        "interface_stats": {
            "total_requests": interface_stats["total_requests"],
            "successful_generations": interface_stats["successful_generations"],
            "success_rate": (interface_stats["successful_generations"] / max(1, interface_stats["total_requests"])) * 100,
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}:{int((uptime % 3600) // 60):02d}:{int(uptime % 60):02d}"
        },
        "system_stats": system_stats
    })

@app.route('/api/gallery')
def get_gallery():
    """معرض الأعمال المولدة"""
    try:
        gallery_items = []
        
        # البحث عن ملفات النتائج
        for filename in os.listdir('.'):
            if filename.startswith('generation_result_') and filename.endswith('.json'):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    if result_data.get('success'):
                        gallery_items.append({
                            "id": filename.replace('generation_result_', '').replace('.json', ''),
                            "shape_name": result_data.get('metadata', {}).get('shape_name', 'غير محدد'),
                            "generated_content": result_data.get('generated_content', {}),
                            "quality_metrics": result_data.get('quality_metrics', {}),
                            "artistic_scores": result_data.get('artistic_scores', {}),
                            "processing_time": result_data.get('processing_time', 0),
                            "creation_date": result_data.get('metadata', {}).get('processing_date', '')
                        })
                except:
                    continue
        
        # ترتيب حسب التاريخ
        gallery_items.sort(key=lambda x: x.get('creation_date', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "gallery": gallery_items[:20],  # أحدث 20 عنصر
            "total": len(gallery_items)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"خطأ في جلب المعرض: {e}"
        })

@app.route('/download/<filename>')
def download_file(filename):
    """تحميل الملفات المولدة"""
    try:
        if os.path.exists(filename):
            return send_file(filename, as_attachment=True)
        else:
            return jsonify({"error": "الملف غير موجود"}), 404
    except Exception as e:
        return jsonify({"error": f"خطأ في التحميل: {e}"}), 500

@app.route('/preview/<filename>')
def preview_file(filename):
    """معاينة الملفات"""
    try:
        if os.path.exists(filename):
            return send_file(filename)
        else:
            return jsonify({"error": "الملف غير موجود"}), 404
    except Exception as e:
        return jsonify({"error": f"خطأ في المعاينة: {e}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "الصفحة غير موجودة"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "خطأ داخلي في الخادم"}), 500

def create_templates_directory():
    """إنشاء مجلد القوالب"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    return templates_dir

def create_static_directory():
    """إنشاء مجلد الملفات الثابتة"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    return static_dir

if __name__ == '__main__':
    print("🌟" + "="*80 + "🌟")
    print("🌐 واجهة الويب للتوليد البصري الثوري")
    print("🎨 نظام بصيرة - إبداع باسل يحيى عبدالله")
    print("🌟" + "="*80 + "🌟")
    
    # إنشاء المجلدات المطلوبة
    create_templates_directory()
    create_static_directory()
    
    print("✅ تم تهيئة واجهة الويب")
    print("🌐 الواجهة متاحة على: http://localhost:5000")
    print("📊 إحصائيات النظام: http://localhost:5000/api/stats")
    print("🎨 معرض الأعمال: http://localhost:5000/api/gallery")
    
    # تشغيل الخادم
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
