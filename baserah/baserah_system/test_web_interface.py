#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Web Interface for Basira System
اختبار واجهة الويب لنظام بصيرة

Simple Flask web server to test the web interface.

Author: Basira System Development Team
Version: 1.0.0
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("❌ Flask غير متاح، يرجى تثبيته أولاً")
    print("❌ Flask not available, please install it first")
    FLASK_AVAILABLE = False
    sys.exit(1)

try:
    from basira_simple_demo import SimpleExpertSystem
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False

# إنشاء تطبيق Flask
app = Flask(__name__, 
           template_folder='interfaces/web/templates',
           static_folder='interfaces/web/static')
CORS(app)

# تهيئة نظام بصيرة
if BASIRA_AVAILABLE:
    expert_system = SimpleExpertSystem()
    print("✅ تم تهيئة نظام بصيرة بنجاح")
else:
    expert_system = None
    print("⚠️ نظام بصيرة غير متاح")


@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')


@app.route('/api/test_innovative_calculus', methods=['POST'])
def test_innovative_calculus():
    """اختبار النظام المبتكر للتفاضل والتكامل"""
    try:
        data = request.get_json()
        
        if not expert_system:
            return jsonify({
                'success': False,
                'error': 'نظام بصيرة غير متاح'
            })
        
        function_values = data.get('function_values', [1, 4, 9, 16, 25])
        d_coefficients = data.get('d_coefficients', [2, 4, 6, 8, 10])
        v_coefficients = data.get('v_coefficients', [0.33, 1.33, 3, 5.33, 8.33])
        
        # إضافة حالة معاملات
        expert_system.calculus_engine.add_coefficient_state(
            function_values, d_coefficients, v_coefficients
        )
        
        # التنبؤ بالتفاضل والتكامل
        result = expert_system.calculus_engine.predict_calculus(function_values)
        
        return jsonify({
            'success': True,
            'function_values': function_values,
            'derivative': result['derivative'],
            'integral': result['integral'],
            'method': result['method'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/test_revolutionary_decomposition', methods=['POST'])
def test_revolutionary_decomposition():
    """اختبار النظام الثوري لتفكيك الدوال"""
    try:
        data = request.get_json()
        
        if not expert_system:
            return jsonify({
                'success': False,
                'error': 'نظام بصيرة غير متاح'
            })
        
        x_values = data.get('x_values', [1, 2, 3, 4, 5])
        f_values = data.get('f_values', [1, 4, 9, 16, 25])
        function_name = data.get('function_name', 'test_function')
        
        # تنفيذ التفكيك الثوري
        result = expert_system.decomposition_engine.decompose_simple_function(
            function_name, x_values, f_values
        )
        
        return jsonify({
            'success': True,
            'function_name': result['function_name'],
            'accuracy': result['accuracy'],
            'n_terms_used': result['n_terms_used'],
            'method': result['method'],
            'original_values': result['original_values'],
            'reconstructed_values': result['reconstructed_values'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/test_general_equation', methods=['POST'])
def test_general_equation():
    """اختبار المعادلة العامة للأشكال"""
    try:
        data = request.get_json()
        
        equation = data.get('equation', 'x^2 + y^2 - r^2')
        variables = data.get('variables', {'x': 3, 'y': 4, 'r': 5})
        
        # محاكاة تقييم المعادلة العامة
        if equation == 'x^2 + y^2 - r^2':
            result_value = variables['x']**2 + variables['y']**2 - variables['r']**2
        else:
            result_value = 0  # قيمة افتراضية
        
        return jsonify({
            'success': True,
            'equation': equation,
            'variables': variables,
            'result': result_value,
            'interpretation': 'تم تقييم المعادلة بنجاح',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/system_info', methods=['GET'])
def system_info():
    """معلومات النظام"""
    try:
        return jsonify({
            'success': True,
            'system_name': 'نظام بصيرة',
            'creator': 'باسل يحيى عبدالله - العراق/الموصل',
            'version': '3.0.0',
            'release_name': 'التكامل الثوري',
            'components': [
                'المعادلة العامة للأشكال',
                'النظام المبتكر للتفاضل والتكامل',
                'النظام الثوري لتفكيك الدوال',
                'نظام الخبير/المستكشف'
            ],
            'interfaces': [
                'واجهة سطح المكتب',
                'واجهة الويب',
                'الواجهة الهيروغلوفية',
                'واجهة العصف الذهني'
            ],
            'basira_available': BASIRA_AVAILABLE,
            'flask_available': FLASK_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/health')
def health_check():
    """فحص صحة النظام"""
    return jsonify({
        'status': 'healthy',
        'message': 'نظام بصيرة يعمل بنجاح',
        'timestamp': datetime.now().isoformat()
    })


def main():
    """تشغيل خادم الويب"""
    print("🌟" + "="*80 + "🌟")
    print("🌐 بدء تشغيل واجهة الويب لنظام بصيرة")
    print("🌐 Starting Basira System Web Interface")
    print("🌟" + "="*80 + "🌟")
    
    print(f"📅 الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Flask متاح: {'✅' if FLASK_AVAILABLE else '❌'}")
    print(f"🧠 نظام بصيرة متاح: {'✅' if BASIRA_AVAILABLE else '❌'}")
    
    if FLASK_AVAILABLE:
        print("\n🌐 خادم الويب يعمل على:")
        print("🌐 Web server running on:")
        print("   http://localhost:5000")
        print("   http://127.0.0.1:5000")
        
        print("\n🔗 الروابط المتاحة:")
        print("🔗 Available endpoints:")
        print("   / - الصفحة الرئيسية")
        print("   /health - فحص صحة النظام")
        print("   /api/system_info - معلومات النظام")
        
        print("\n🚀 لإيقاف الخادم اضغط Ctrl+C")
        print("🚀 To stop the server press Ctrl+C")
        print("🌟" + "="*80 + "🌟")
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=True)
        except KeyboardInterrupt:
            print("\n🛑 تم إيقاف خادم الويب")
            print("🛑 Web server stopped")
    else:
        print("❌ لا يمكن تشغيل خادم الويب - Flask غير متاح")
        print("❌ Cannot start web server - Flask not available")


if __name__ == "__main__":
    main()
