#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
واجهة الويب لنظام بصيرة

هذا الملف يحدد واجهة الويب لنظام بصيرة، باستخدام Flask كإطار عمل للتطبيق.
يوفر واجهة تفاعلية متكاملة مع النظام المعرفي والتوليدي الأساسي.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# استيراد من الوحدات الأخرى
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cognitive_linguistic_architecture import CognitiveLinguisticArchitecture
from generative_language_model import SimpleGenerativeLanguageModel, GenerationContext, GenerationMode, ContextLevel
from knowledge_extraction_generation import KnowledgeExtractor, KnowledgeGenerator
from self_learning_adaptive_evolution import AdaptiveEvolutionManager
from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer
from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer
from code_execution.executor import CodeExecutor

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_web_interface')

# تهيئة تطبيق Flask
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.secret_key = os.urandom(24)
socketio = SocketIO(app)

# تكوين مجلد التحميل
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'py', 'js', 'html', 'css'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# تهيئة مكونات النظام
architecture = CognitiveLinguisticArchitecture()
language_model = SimpleGenerativeLanguageModel()
knowledge_extractor = KnowledgeExtractor()
knowledge_generator = KnowledgeGenerator()
evolution_manager = AdaptiveEvolutionManager()
root_extractor = ArabicRootExtractor()
syntax_analyzer = ArabicSyntaxAnalyzer()
rhetoric_analyzer = ArabicRhetoricAnalyzer()
code_executor = CodeExecutor()

# حالة النظام
system_state = {
    'is_initialized': False,
    'is_learning': False,
    'is_generating': False,
    'current_context': None,
    'last_generation': None,
    'active_concepts': [],
    'execution_results': [],
    'chat_history': []
}


def allowed_file(filename: str) -> bool:
    """
    التحقق من أن الملف له امتداد مسموح به.
    
    Args:
        filename: اسم الملف
        
    Returns:
        True إذا كان الملف مسموحاً به، وإلا False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """الصفحة الرئيسية."""
    return render_template('index.html')


@app.route('/chat')
def chat():
    """صفحة المحادثة."""
    return render_template('chat.html', history=system_state['chat_history'])


@app.route('/knowledge')
def knowledge():
    """صفحة استعراض المعرفة."""
    return render_template('knowledge.html', concepts=system_state['active_concepts'])


@app.route('/generate')
def generate():
    """صفحة التوليد."""
    return render_template('generate.html', last_generation=system_state['last_generation'])


@app.route('/code')
def code():
    """صفحة تنفيذ الأكواد."""
    return render_template('code.html', results=system_state['execution_results'])


@app.route('/settings')
def settings():
    """صفحة الإعدادات."""
    return render_template('settings.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    واجهة برمجة التطبيقات للمحادثة.
    
    Returns:
        استجابة JSON تحتوي على رد النظام
    """
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # إضافة رسالة المستخدم إلى سجل المحادثة
    system_state['chat_history'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': 'now'  # يمكن استبدالها بطابع زمني حقيقي
    })
    
    # إنشاء سياق التوليد
    context = GenerationContext(
        context_level=ContextLevel.DIALOGUE,
        generation_mode=GenerationMode.HYBRID
    )
    
    # توليد الرد
    system_response = language_model.generate(context, max_length=500)
    
    # إضافة رد النظام إلى سجل المحادثة
    system_state['chat_history'].append({
        'role': 'system',
        'content': system_response,
        'timestamp': 'now'  # يمكن استبدالها بطابع زمني حقيقي
    })
    
    return jsonify({
        'response': system_response,
        'context': context.to_dict() if hasattr(context, 'to_dict') else None
    })


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    واجهة برمجة التطبيقات للتوليد.
    
    Returns:
        استجابة JSON تحتوي على النص المولد
    """
    data = request.json
    prompt = data.get('prompt', '')
    mode = data.get('mode', 'HYBRID')
    level = data.get('level', 'PARAGRAPH')
    max_length = int(data.get('max_length', 500))
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # إنشاء سياق التوليد
    context = GenerationContext(
        context_level=getattr(ContextLevel, level, ContextLevel.PARAGRAPH),
        generation_mode=getattr(GenerationMode, mode, GenerationMode.HYBRID)
    )
    
    # توليد النص
    generated_text = language_model.generate(context, max_length=max_length)
    
    # تحديث حالة النظام
    system_state['last_generation'] = generated_text
    system_state['current_context'] = context.to_dict() if hasattr(context, 'to_dict') else None
    
    return jsonify({
        'generated_text': generated_text,
        'context': context.to_dict() if hasattr(context, 'to_dict') else None
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    واجهة برمجة التطبيقات لتحليل النص.
    
    Returns:
        استجابة JSON تحتوي على نتائج التحليل
    """
    data = request.json
    text = data.get('text', '')
    analysis_type = data.get('analysis_type', 'all')  # morphology, syntax, rhetoric, all
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    results = {}
    
    # تحليل صرفي
    if analysis_type in ['morphology', 'all']:
        morphology_results = {
            'roots': root_extractor.extract_roots(text),
            'stems': root_extractor.extract_stems(text),
            'patterns': root_extractor.identify_patterns(text)
        }
        results['morphology'] = morphology_results
    
    # تحليل نحوي
    if analysis_type in ['syntax', 'all']:
        syntax_results = {
            'pos_tags': syntax_analyzer.pos_tag(text),
            'parse_tree': syntax_analyzer.parse(text),
            'dependencies': syntax_analyzer.dependency_parse(text)
        }
        results['syntax'] = syntax_results
    
    # تحليل بلاغي
    if analysis_type in ['rhetoric', 'all']:
        rhetoric_results = {
            'figures': rhetoric_analyzer.identify_figures(text),
            'style': rhetoric_analyzer.analyze_style(text),
            'aesthetics': rhetoric_analyzer.evaluate_aesthetics(text)
        }
        results['rhetoric'] = rhetoric_results
    
    return jsonify(results)


@app.route('/api/execute_code', methods=['POST'])
def api_execute_code():
    """
    واجهة برمجة التطبيقات لتنفيذ الأكواد.
    
    Returns:
        استجابة JSON تحتوي على نتائج التنفيذ
    """
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    # تنفيذ الكود
    result = code_executor.execute(code, language)
    
    # تحديث حالة النظام
    system_state['execution_results'].append({
        'code': code,
        'language': language,
        'result': result,
        'timestamp': 'now'  # يمكن استبدالها بطابع زمني حقيقي
    })
    
    return jsonify(result)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """
    واجهة برمجة التطبيقات لتحميل الملفات.
    
    Returns:
        استجابة JSON تحتوي على معلومات الملف المحمل
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'path': file_path
        })
    
    return jsonify({'error': 'File type not allowed'}), 400


@socketio.on('connect')
def handle_connect():
    """معالجة اتصال Socket.IO."""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to Basira system'})


@socketio.on('disconnect')
def handle_disconnect():
    """معالجة قطع اتصال Socket.IO."""
    logger.info('Client disconnected')


@socketio.on('stream_generate')
def handle_stream_generate(data):
    """
    معالجة طلب توليد متدفق.
    
    Args:
        data: بيانات الطلب
    """
    prompt = data.get('prompt', '')
    mode = data.get('mode', 'HYBRID')
    level = data.get('level', 'PARAGRAPH')
    max_length = int(data.get('max_length', 500))
    
    if not prompt:
        emit('error', {'message': 'No prompt provided'})
        return
    
    # إنشاء سياق التوليد
    context = GenerationContext(
        context_level=getattr(ContextLevel, level, ContextLevel.PARAGRAPH),
        generation_mode=getattr(GenerationMode, mode, GenerationMode.HYBRID)
    )
    
    # تعيين حالة التوليد
    system_state['is_generating'] = True
    
    # محاكاة التوليد المتدفق
    generated_text = ""
    for i in range(10):  # محاكاة 10 دفعات من التوليد
        # توليد جزء من النص
        chunk = language_model.generate(context, max_length=max_length // 10)
        generated_text += chunk
        
        # إرسال الجزء المولد
        emit('generation_chunk', {
            'chunk': chunk,
            'is_final': i == 9
        })
    
    # تحديث حالة النظام
    system_state['is_generating'] = False
    system_state['last_generation'] = generated_text
    system_state['current_context'] = context.to_dict() if hasattr(context, 'to_dict') else None


if __name__ == '__main__':
    # تهيئة النظام
    system_state['is_initialized'] = True
    
    # تشغيل التطبيق
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
