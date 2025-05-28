#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Web Server for Basira System

This module implements a simple web server for the Basira System using
Python's built-in http.server module.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from main import BasiraSystem
    from dream_interpretation.basira_dream_integration import create_basira_dream_system
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('basira_web_server')


class BasiraWebHandler(BaseHTTPRequestHandler):
    """معالج طلبات الويب لنظام بصيرة"""
    
    def __init__(self, *args, **kwargs):
        self.basira_system = None
        self.dream_system = None
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """معالجة طلبات GET"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_main_page()
        elif path == '/dream':
            self.serve_dream_page()
        elif path == '/code':
            self.serve_code_page()
        elif path == '/image':
            self.serve_image_page()
        elif path == '/status':
            self.serve_status_page()
        elif path.startswith('/api/'):
            self.handle_api_request(path, parsed_path.query)
        else:
            self.send_error(404, "الصفحة غير موجودة")
    
    def do_POST(self):
        """معالجة طلبات POST"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path.startswith('/api/'):
            # قراءة البيانات المرسلة
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
            except:
                data = {}
            
            self.handle_api_post(path, data)
        else:
            self.send_error(405, "الطريقة غير مدعومة")
    
    def serve_main_page(self):
        """عرض الصفحة الرئيسية"""
        html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام بصيرة - الواجهة الرئيسية</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            color: white;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            font-size: 1.2em;
            margin: 10px 0;
            opacity: 0.9;
        }}
        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }}
        .feature-card {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: transform 0.3s ease;
            cursor: pointer;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.3);
        }}
        .feature-icon {{
            font-size: 3em;
            margin-bottom: 15px;
        }}
        .feature-title {{
            font-size: 1.5em;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        .feature-desc {{
            opacity: 0.9;
            line-height: 1.6;
        }}
        .status-bar {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin-top: 30px;
            text-align: center;
        }}
        .btn {{
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            border: none;
            border-radius: 25px;
            padding: 12px 25px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }}
        .btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 نظام بصيرة 🌟</h1>
            <p>نظام ذكاء اصطناعي متكامل يجمع بين التراث والحداثة</p>
            <p>حيث تلتقي الحكمة القديمة بالتقنية المتطورة</p>
        </div>
        
        <div class="features">
            <div class="feature-card" onclick="window.location.href='/dream'">
                <div class="feature-icon">🌙</div>
                <div class="feature-title">تفسير الأحلام</div>
                <div class="feature-desc">
                    تفسير الأحلام وفق نظرية "العقل النعسان" المبتكرة
                    مع آليات الترميز المتقدمة والتفسير متعدد الطبقات
                </div>
            </div>
            
            <div class="feature-card" onclick="window.location.href='/code'">
                <div class="feature-icon">💻</div>
                <div class="feature-title">تنفيذ الأكواد</div>
                <div class="feature-desc">
                    تنفيذ آمن للأكواد بلغات برمجة متعددة
                    مع بيئة معزولة وحدود أمان متقدمة
                </div>
            </div>
            
            <div class="feature-card" onclick="window.location.href='/image'">
                <div class="feature-icon">🎨</div>
                <div class="feature-title">توليد الصور</div>
                <div class="feature-desc">
                    توليد الصور من النصوص العربية والإنجليزية
                    مع تقنيات الذكاء الاصطناعي المتطورة
                </div>
            </div>
            
            <div class="feature-card" onclick="showComingSoon()">
                <div class="feature-icon">🎬</div>
                <div class="feature-title">توليد الفيديو</div>
                <div class="feature-desc">
                    توليد مقاطع الفيديو والرسوم المتحركة
                    من النصوص والمفاهيم
                </div>
            </div>
            
            <div class="feature-card" onclick="showComingSoon()">
                <div class="feature-icon">📝</div>
                <div class="feature-title">معالجة العربية</div>
                <div class="feature-desc">
                    تحليل متقدم للنصوص العربية مع استخراج الجذور
                    والتحليل الصرفي والنحوي والبلاغي
                </div>
            </div>
            
            <div class="feature-card" onclick="showComingSoon()">
                <div class="feature-icon">🧮</div>
                <div class="feature-title">حل المعادلات</div>
                <div class="feature-desc">
                    حل المعادلات الرياضية والألغاز المنطقية
                    باستخدام المعادلة العامة للأشكال
                </div>
            </div>
            
            <div class="feature-card" onclick="showComingSoon()">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">العصف الذهني</div>
                <div class="feature-desc">
                    خرائط ذهنية تفاعلية تُظهر ترابط الأفكار
                    والمعلومات بطريقة بصرية مبتكرة
                </div>
            </div>
            
            <div class="feature-card" onclick="window.location.href='/status'">
                <div class="feature-icon">📊</div>
                <div class="feature-title">مراقبة النظام</div>
                <div class="feature-desc">
                    مراقبة حالة النظام والمكونات المختلفة
                    مع إحصائيات الأداء والاستخدام
                </div>
            </div>
        </div>
        
        <div class="status-bar">
            <strong>حالة النظام:</strong> 
            <span id="system-status">جاري التحقق...</span>
            <br>
            <strong>الوقت:</strong> 
            <span id="current-time">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            <br><br>
            <a href="/status" class="btn">عرض التفاصيل</a>
            <a href="javascript:location.reload()" class="btn">تحديث</a>
        </div>
    </div>
    
    <script>
        function showComingSoon() {{
            alert('🚧 هذه الميزة قيد التطوير وستكون متاحة قريباً! 🚧');
        }}
        
        function updateTime() {{
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString('ar-SA');
        }}
        
        function checkSystemStatus() {{
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {{
                    const statusElement = document.getElementById('system-status');
                    if (data.basira_available) {{
                        statusElement.textContent = '✅ النظام يعمل بشكل ممتاز';
                        statusElement.style.color = '#4CAF50';
                    }} else {{
                        statusElement.textContent = '⚠️ النظام يعمل جزئياً';
                        statusElement.style.color = '#FF9800';
                    }}
                }})
                .catch(error => {{
                    document.getElementById('system-status').textContent = '❌ خطأ في الاتصال';
                    document.getElementById('system-status').style.color = '#F44336';
                }});
        }}
        
        // تحديث الوقت كل ثانية
        setInterval(updateTime, 1000);
        
        // فحص حالة النظام عند تحميل الصفحة
        checkSystemStatus();
        
        // تحديث حالة النظام كل 30 ثانية
        setInterval(checkSystemStatus, 30000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_dream_page(self):
        """عرض صفحة تفسير الأحلام"""
        html = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تفسير الأحلام - نظام بصيرة</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            margin: 0;
            padding: 20px;
            color: white;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        textarea, input, select {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 16px;
        }
        textarea::placeholder, input::placeholder {
            color: rgba(255, 255, 255, 0.7);
        }
        .btn {
            background: linear-gradient(45deg, #e74c3c, #f39c12);
            border: none;
            border-radius: 25px;
            padding: 15px 30px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin: 10px 0;
        }
        .btn:hover {
            transform: scale(1.02);
        }
        .result {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .back-btn {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            border-radius: 20px;
            padding: 10px 20px;
            color: white;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← العودة للرئيسية</a>
        
        <div class="header">
            <h1>🌙 تفسير الأحلام</h1>
            <p>تفسير الأحلام وفق نظرية "العقل النعسان" المبتكرة</p>
        </div>
        
        <form id="dreamForm">
            <div class="form-group">
                <label for="dreamText">نص الحلم:</label>
                <textarea id="dreamText" rows="5" placeholder="اكتب نص الحلم هنا..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="dreamerName">اسم الرائي:</label>
                <input type="text" id="dreamerName" placeholder="الاسم (اختياري)">
            </div>
            
            <div class="form-group">
                <label for="dreamerProfession">المهنة:</label>
                <input type="text" id="dreamerProfession" placeholder="المهنة (اختياري)">
            </div>
            
            <div class="form-group">
                <label for="dreamerReligion">الديانة:</label>
                <select id="dreamerReligion">
                    <option value="إسلام">إسلام</option>
                    <option value="مسيحية">مسيحية</option>
                    <option value="يهودية">يهودية</option>
                    <option value="أخرى">أخرى</option>
                </select>
            </div>
            
            <button type="submit" class="btn">🔍 فسر الحلم</button>
        </form>
        
        <div id="result" class="result">
            <h3>نتيجة التفسير:</h3>
            <div id="interpretation"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('dreamForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const dreamText = document.getElementById('dreamText').value;
            const dreamerName = document.getElementById('dreamerName').value;
            const dreamerProfession = document.getElementById('dreamerProfession').value;
            const dreamerReligion = document.getElementById('dreamerReligion').value;
            
            if (!dreamText.trim()) {
                alert('يرجى إدخال نص الحلم');
                return;
            }
            
            const data = {
                dream_text: dreamText,
                user_info: {
                    name: dreamerName || 'غير محدد',
                    profession: dreamerProfession || 'غير محدد',
                    religion: dreamerReligion
                }
            };
            
            // إظهار رسالة التحميل
            document.getElementById('result').style.display = 'block';
            document.getElementById('interpretation').innerHTML = '<p>🔄 جاري تفسير الحلم...</p>';
            
            fetch('/api/interpret_dream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    let html = `
                        <h4>📊 معلومات التفسير:</h4>
                        <p><strong>نوع الحلم:</strong> ${data.basic_interpretation.dream_type}</p>
                        <p><strong>مستوى الثقة:</strong> ${(data.basic_interpretation.confidence_level * 100).toFixed(0)}%</p>
                        
                        <h4>💭 التفسير:</h4>
                        <p>${data.basic_interpretation.overall_message}</p>
                        
                        <h4>📋 التوصيات:</h4>
                        <ul>
                    `;
                    
                    data.recommendations.forEach(rec => {
                        html += `<li>${rec}</li>`;
                    });
                    
                    html += '</ul>';
                    
                    if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                        html += '<h4>❓ أسئلة المتابعة:</h4><ul>';
                        data.follow_up_questions.forEach(q => {
                            html += `<li>${q}</li>`;
                        });
                        html += '</ul>';
                    }
                    
                    document.getElementById('interpretation').innerHTML = html;
                } else {
                    document.getElementById('interpretation').innerHTML = `<p style="color: #e74c3c;">❌ خطأ: ${data.error}</p>`;
                }
            })
            .catch(error => {
                document.getElementById('interpretation').innerHTML = `<p style="color: #e74c3c;">❌ خطأ في الاتصال: ${error.message}</p>`;
            });
        });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def handle_api_request(self, path, query):
        """معالجة طلبات API"""
        if path == '/api/status':
            self.send_json_response({
                'basira_available': BASIRA_AVAILABLE,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'dream_interpretation': BASIRA_AVAILABLE,
                    'code_execution': BASIRA_AVAILABLE,
                    'image_generation': BASIRA_AVAILABLE
                }
            })
        else:
            self.send_error(404, "API endpoint not found")
    
    def handle_api_post(self, path, data):
        """معالجة طلبات POST للـ API"""
        if path == '/api/interpret_dream':
            self.interpret_dream_api(data)
        else:
            self.send_error(404, "API endpoint not found")
    
    def interpret_dream_api(self, data):
        """API تفسير الأحلام"""
        try:
            dream_text = data.get('dream_text', '')
            user_info = data.get('user_info', {})
            
            if not dream_text:
                self.send_json_response({'success': False, 'error': 'نص الحلم مطلوب'})
                return
            
            # محاكاة تفسير الحلم
            result = {
                'success': True,
                'session_id': f'web_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'basic_interpretation': {
                    'dream_type': 'رؤيا_صادقة',
                    'confidence_level': 0.85,
                    'overall_message': f'الحلم "{dream_text}" يحمل دلالات إيجابية ويشير إلى الخير والبركة. العناصر المذكورة في الحلم تدل على النمو والتطور الروحي.'
                },
                'recommendations': [
                    'احرص على الصلاة والذكر',
                    'تفاءل بالخير واستبشر',
                    'اعمل على تطوير نفسك',
                    'كن صبوراً في تحقيق أهدافك',
                    'اطلب الهداية من الله'
                ],
                'follow_up_questions': [
                    'هل رأيت ألواناً معينة في الحلم؟',
                    'ما هو شعورك أثناء الحلم؟',
                    'هل هناك أشخاص معروفون في الحلم؟'
                ]
            }
            
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({'success': False, 'error': str(e)})
    
    def send_json_response(self, data):
        """إرسال استجابة JSON"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def log_message(self, format, *args):
        """تسجيل الرسائل"""
        logger.info(f"{self.address_string()} - {format % args}")


def start_web_server(port=8000):
    """بدء خادم الويب"""
    try:
        server_address = ('', port)
        httpd = HTTPServer(server_address, BasiraWebHandler)
        
        print(f"🌐 خادم الويب يعمل على المنفذ {port}")
        print(f"🔗 افتح المتصفح على: http://localhost:{port}")
        print("⏹️ اضغط Ctrl+C لإيقاف الخادم")
        
        # فتح المتصفح تلقائياً
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
        
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف الخادم")
        httpd.shutdown()
    except Exception as e:
        print(f"❌ خطأ في تشغيل الخادم: {e}")


if __name__ == "__main__":
    start_web_server()
