#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basira Desktop Interface - Advanced GUI Application

This module implements a comprehensive desktop interface for the Basira System
using tkinter with modern styling and advanced features.

Author: Basira System Development Team
Version: 2.0.0
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sys
import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from main import BasiraSystem
    from dream_interpretation.basira_dream_integration import create_basira_dream_system
    from code_execution.code_executor import CodeExecutor, ProgrammingLanguage
    from creative_generation.image.image_generator import ImageGenerator, GenerationParameters, GenerationMode
    from arabic_nlp.advanced_processor import ArabicNLPProcessor
    from mathematical_core.general_shape_equation import GeneralShapeEquation, EquationType
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False


class BasiraDesktopApp:
    """تطبيق سطح المكتب الرئيسي لنظام بصيرة"""

    def __init__(self):
        """تهيئة التطبيق"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - واجهة سطح المكتب")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # تهيئة مكونات النظام
        self.basira_system = None
        self.dream_system = None
        self.code_executor = None
        self.image_generator = None
        self.arabic_processor = None

        # إنشاء الواجهة
        self.create_widgets()
        self.initialize_basira_components()

    def create_widgets(self):
        """إنشاء عناصر الواجهة"""
        # شريط القوائم
        self.create_menu_bar()

        # الشريط الجانبي
        self.create_sidebar()

        # المنطقة الرئيسية
        self.create_main_area()

        # شريط الحالة
        self.create_status_bar()

    def create_menu_bar(self):
        """إنشاء شريط القوائم"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # قائمة النظام
        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="النظام", menu=system_menu)
        system_menu.add_command(label="حالة النظام", command=self.show_system_status)
        system_menu.add_command(label="إعادة تشغيل", command=self.restart_system)
        system_menu.add_separator()
        system_menu.add_command(label="خروج", command=self.root.quit)

        # قائمة الأدوات
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="الأدوات", menu=tools_menu)
        tools_menu.add_command(label="تفسير الأحلام", command=self.show_dream_tab)
        tools_menu.add_command(label="تنفيذ الكود", command=self.show_code_tab)
        tools_menu.add_command(label="توليد الصور", command=self.show_image_tab)
        tools_menu.add_command(label="معالجة العربية", command=self.show_arabic_tab)
        tools_menu.add_command(label="حل المعادلات", command=self.show_math_tab)

        # قائمة المساعدة
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="مساعدة", menu=help_menu)
        help_menu.add_command(label="حول النظام", command=self.show_about)
        help_menu.add_command(label="دليل المستخدم", command=self.show_help)

    def create_sidebar(self):
        """إنشاء الشريط الجانبي"""
        sidebar_frame = ttk.Frame(self.root, width=200)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        sidebar_frame.pack_propagate(False)

        # عنوان الشريط الجانبي
        title_label = ttk.Label(sidebar_frame, text="🌟 نظام بصيرة",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # أزرار الوظائف الرئيسية
        buttons = [
            ("🌙 تفسير الأحلام", self.show_dream_tab),
            ("💻 تنفيذ الكود", self.show_code_tab),
            ("🎨 توليد الصور", self.show_image_tab),
            ("🎬 توليد الفيديو", self.show_video_tab),
            ("📝 معالجة العربية", self.show_arabic_tab),
            ("🧮 حل المعادلات", self.show_math_tab),
            ("🧠 العصف الذهني", self.show_brainstorm_tab),
            ("📊 مراقبة النظام", self.show_monitor_tab)
        ]

        for text, command in buttons:
            btn = ttk.Button(sidebar_frame, text=text, command=command, width=20)
            btn.pack(pady=2, padx=5, fill=tk.X)

        # معلومات النظام
        info_frame = ttk.LabelFrame(sidebar_frame, text="معلومات النظام")
        info_frame.pack(pady=10, padx=5, fill=tk.X)

        self.status_label = ttk.Label(info_frame, text="جاري التحميل...")
        self.status_label.pack(pady=5)

        self.time_label = ttk.Label(info_frame, text="")
        self.time_label.pack(pady=2)

        # تحديث الوقت
        self.update_time()

    def create_main_area(self):
        """إنشاء المنطقة الرئيسية"""
        # إطار رئيسي
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # دفتر التبويبات
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # تبويب الترحيب
        self.create_welcome_tab()

        # تبويبات الوظائف
        self.create_dream_tab()
        self.create_code_tab()
        self.create_image_tab()
        self.create_video_tab()
        self.create_arabic_tab()
        self.create_math_tab()
        self.create_brainstorm_tab()
        self.create_monitor_tab()

    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.StringVar()
        self.status_text.set("مرحباً بك في نظام بصيرة")

        status_label = ttk.Label(status_frame, textvariable=self.status_text)
        status_label.pack(side=tk.LEFT, padx=5)

        # مؤشر التقدم
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=5)

    def create_welcome_tab(self):
        """إنشاء تبويب الترحيب"""
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="🏠 الرئيسية")

        # عنوان الترحيب
        welcome_title = ttk.Label(welcome_frame,
                                 text="مرحباً بك في نظام بصيرة",
                                 font=('Arial', 20, 'bold'))
        welcome_title.pack(pady=20)

        # وصف النظام
        description = """
        نظام بصيرة هو نظام ذكاء اصطناعي متكامل ومبتكر يجمع بين التراث العربي الإسلامي والتقنيات الحديثة

        🌟 المميزات الرئيسية:
        • تفسير الأحلام وفق نظرية "العقل النعسان"
        • تنفيذ الأكواد بلغات برمجة متعددة
        • توليد الصور والفيديوهات من النصوص
        • معالجة متقدمة للغة العربية
        • حل المعادلات والألغاز الرياضية
        • العصف الذهني وربط المعلومات

        🚀 ابدأ باختيار إحدى الوظائف من الشريط الجانبي
        """

        desc_label = ttk.Label(welcome_frame, text=description,
                              font=('Arial', 12), justify=tk.CENTER)
        desc_label.pack(pady=20, padx=20)

        # أزرار سريعة
        quick_frame = ttk.Frame(welcome_frame)
        quick_frame.pack(pady=20)

        quick_buttons = [
            ("🌙 تفسير حلم", self.show_dream_tab),
            ("💻 تشغيل كود", self.show_code_tab),
            ("🎨 إنشاء صورة", self.show_image_tab)
        ]

        for text, command in quick_buttons:
            btn = ttk.Button(quick_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=10)

    def create_dream_tab(self):
        """إنشاء تبويب تفسير الأحلام"""
        dream_frame = ttk.Frame(self.notebook)
        self.notebook.add(dream_frame, text="🌙 تفسير الأحلام")

        # إطار الإدخال
        input_frame = ttk.LabelFrame(dream_frame, text="أدخل نص الحلم")
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.dream_text = scrolledtext.ScrolledText(input_frame, height=5,
                                                   font=('Arial', 12))
        self.dream_text.pack(fill=tk.X, padx=5, pady=5)

        # إطار معلومات الرائي
        user_frame = ttk.LabelFrame(dream_frame, text="معلومات الرائي")
        user_frame.pack(fill=tk.X, padx=10, pady=5)

        # حقول المعلومات
        fields_frame = ttk.Frame(user_frame)
        fields_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(fields_frame, text="الاسم:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dreamer_name = ttk.Entry(fields_frame, width=20)
        self.dreamer_name.grid(row=0, column=1, padx=5)

        ttk.Label(fields_frame, text="المهنة:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.dreamer_profession = ttk.Entry(fields_frame, width=20)
        self.dreamer_profession.grid(row=0, column=3, padx=5)

        ttk.Label(fields_frame, text="الديانة:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.dreamer_religion = ttk.Combobox(fields_frame, values=["إسلام", "مسيحية", "يهودية", "أخرى"])
        self.dreamer_religion.grid(row=1, column=1, padx=5)
        self.dreamer_religion.set("إسلام")

        # زر التفسير
        interpret_btn = ttk.Button(dream_frame, text="🔍 فسر الحلم",
                                  command=self.interpret_dream)
        interpret_btn.pack(pady=10)

        # إطار النتائج
        result_frame = ttk.LabelFrame(dream_frame, text="نتيجة التفسير")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.dream_result = scrolledtext.ScrolledText(result_frame,
                                                     font=('Arial', 11))
        self.dream_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_code_tab(self):
        """إنشاء تبويب تنفيذ الكود"""
        code_frame = ttk.Frame(self.notebook)
        self.notebook.add(code_frame, text="💻 تنفيذ الكود")

        # إطار اختيار اللغة
        lang_frame = ttk.Frame(code_frame)
        lang_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(lang_frame, text="لغة البرمجة:").pack(side=tk.LEFT)
        self.code_language = ttk.Combobox(lang_frame,
                                         values=["python", "javascript", "bash"])
        self.code_language.pack(side=tk.LEFT, padx=5)
        self.code_language.set("python")

        run_btn = ttk.Button(lang_frame, text="▶️ تشغيل", command=self.execute_code)
        run_btn.pack(side=tk.RIGHT)

        # إطار الكود
        code_input_frame = ttk.LabelFrame(code_frame, text="الكود")
        code_input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.code_text = scrolledtext.ScrolledText(code_input_frame,
                                                  font=('Courier', 11))
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # إطار النتائج
        output_frame = ttk.LabelFrame(code_frame, text="النتيجة")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.code_output = scrolledtext.ScrolledText(output_frame,
                                                    font=('Courier', 10))
        self.code_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_image_tab(self):
        """إنشاء تبويب توليد الصور"""
        image_frame = ttk.Frame(self.notebook)
        self.notebook.add(image_frame, text="🎨 توليد الصور")

        # إطار الإدخال
        input_frame = ttk.LabelFrame(image_frame, text="وصف الصورة")
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.image_prompt = scrolledtext.ScrolledText(input_frame, height=3)
        self.image_prompt.pack(fill=tk.X, padx=5, pady=5)

        # إطار الإعدادات
        settings_frame = ttk.LabelFrame(image_frame, text="إعدادات التوليد")
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # الأبعاد
        dims_frame = ttk.Frame(settings_frame)
        dims_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(dims_frame, text="العرض:").pack(side=tk.LEFT)
        self.image_width = ttk.Spinbox(dims_frame, from_=256, to=1024, value=512)
        self.image_width.pack(side=tk.LEFT, padx=5)

        ttk.Label(dims_frame, text="الارتفاع:").pack(side=tk.LEFT, padx=(10,0))
        self.image_height = ttk.Spinbox(dims_frame, from_=256, to=1024, value=512)
        self.image_height.pack(side=tk.LEFT, padx=5)

        generate_btn = ttk.Button(dims_frame, text="🎨 إنشاء الصورة",
                                 command=self.generate_image)
        generate_btn.pack(side=tk.RIGHT)

        # إطار النتيجة
        result_frame = ttk.LabelFrame(image_frame, text="الصورة المولدة")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.image_result = ttk.Label(result_frame, text="لم يتم إنشاء صورة بعد")
        self.image_result.pack(expand=True)

    def create_video_tab(self):
        """إنشاء تبويب توليد الفيديو"""
        video_frame = ttk.Frame(self.notebook)
        self.notebook.add(video_frame, text="🎬 توليد الفيديو")

        ttk.Label(video_frame, text="🎬 توليد الفيديو",
                 font=('Arial', 16, 'bold')).pack(pady=20)
        ttk.Label(video_frame, text="قريباً... سيتم إضافة واجهة توليد الفيديو").pack()

    def create_arabic_tab(self):
        """إنشاء تبويب معالجة العربية"""
        arabic_frame = ttk.Frame(self.notebook)
        self.notebook.add(arabic_frame, text="📝 معالجة العربية")

        # إطار الإدخال
        input_frame = ttk.LabelFrame(arabic_frame, text="النص العربي")
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.arabic_text = scrolledtext.ScrolledText(input_frame, height=5)
        self.arabic_text.pack(fill=tk.X, padx=5, pady=5)

        process_btn = ttk.Button(arabic_frame, text="🔍 تحليل النص",
                               command=self.process_arabic)
        process_btn.pack(pady=10)

        # إطار النتائج
        result_frame = ttk.LabelFrame(arabic_frame, text="نتيجة التحليل")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.arabic_result = scrolledtext.ScrolledText(result_frame)
        self.arabic_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_math_tab(self):
        """إنشاء تبويب حل المعادلات"""
        math_frame = ttk.Frame(self.notebook)
        self.notebook.add(math_frame, text="🧮 حل المعادلات")

        # إطار الإدخال
        input_frame = ttk.LabelFrame(math_frame, text="المعادلة")
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.equation_text = ttk.Entry(input_frame, font=('Arial', 12))
        self.equation_text.pack(fill=tk.X, padx=5, pady=5)

        solve_btn = ttk.Button(math_frame, text="🧮 حل المعادلة",
                              command=self.solve_equation)
        solve_btn.pack(pady=10)

        # إطار النتائج
        result_frame = ttk.LabelFrame(math_frame, text="الحل")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.math_result = scrolledtext.ScrolledText(result_frame)
        self.math_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_brainstorm_tab(self):
        """إنشاء تبويب العصف الذهني"""
        brainstorm_frame = ttk.Frame(self.notebook)
        self.notebook.add(brainstorm_frame, text="🧠 العصف الذهني")

        ttk.Label(brainstorm_frame, text="🧠 خريطة العصف الذهني",
                 font=('Arial', 16, 'bold')).pack(pady=20)

        # إطار الموضوع
        topic_frame = ttk.LabelFrame(brainstorm_frame, text="الموضوع الرئيسي")
        topic_frame.pack(fill=tk.X, padx=10, pady=5)

        self.brainstorm_topic = ttk.Entry(topic_frame, font=('Arial', 12))
        self.brainstorm_topic.pack(fill=tk.X, padx=5, pady=5)

        generate_map_btn = ttk.Button(brainstorm_frame, text="🗺️ إنشاء خريطة ذهنية",
                                     command=self.generate_mind_map)
        generate_map_btn.pack(pady=10)

        # إطار الخريطة الذهنية
        map_frame = ttk.LabelFrame(brainstorm_frame, text="الخريطة الذهنية")
        map_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.mind_map_canvas = tk.Canvas(map_frame, bg='white')
        self.mind_map_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_monitor_tab(self):
        """إنشاء تبويب مراقبة النظام"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="📊 مراقبة النظام")

        # معلومات النظام
        info_frame = ttk.LabelFrame(monitor_frame, text="معلومات النظام")
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.system_info = scrolledtext.ScrolledText(info_frame, height=10)
        self.system_info.pack(fill=tk.X, padx=5, pady=5)

        refresh_btn = ttk.Button(monitor_frame, text="🔄 تحديث",
                               command=self.refresh_system_info)
        refresh_btn.pack(pady=10)

    # وظائف النظام
    def initialize_basira_components(self):
        """تهيئة مكونات نظام بصيرة"""
        def init_in_thread():
            try:
                self.status_text.set("جاري تهيئة مكونات النظام...")
                self.progress.start()

                if BASIRA_AVAILABLE:
                    self.basira_system = BasiraSystem()
                    self.dream_system = create_basira_dream_system()
                    self.code_executor = CodeExecutor()
                    self.image_generator = ImageGenerator()
                    self.arabic_processor = ArabicNLPProcessor()

                    self.status_text.set("✅ تم تهيئة جميع المكونات بنجاح")
                    self.status_label.config(text="✅ النظام جاهز")
                else:
                    self.status_text.set("⚠️ بعض المكونات غير متاحة")
                    self.status_label.config(text="⚠️ جزئي")

            except Exception as e:
                self.status_text.set(f"❌ خطأ في التهيئة: {str(e)}")
                self.status_label.config(text="❌ خطأ")
            finally:
                self.progress.stop()

        threading.Thread(target=init_in_thread, daemon=True).start()

    def interpret_dream(self):
        """تفسير الحلم"""
        dream_text = self.dream_text.get(1.0, tk.END).strip()
        if not dream_text:
            messagebox.showwarning("تحذير", "يرجى إدخال نص الحلم")
            return

        def interpret_in_thread():
            try:
                self.status_text.set("جاري تفسير الحلم...")
                self.progress.start()

                if self.dream_system:
                    # إنشاء ملف المستخدم
                    user_info = {
                        'name': self.dreamer_name.get() or 'مستخدم',
                        'profession': self.dreamer_profession.get() or 'غير محدد',
                        'religion': self.dreamer_religion.get() or 'إسلام'
                    }

                    user_profile = self.dream_system.create_user_profile(
                        f"desktop_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        user_info
                    )

                    # تفسير الحلم
                    result = self.dream_system.interpret_dream_comprehensive(
                        user_profile.user_id, dream_text
                    )

                    # عرض النتيجة
                    if result['success']:
                        interpretation_text = f"""
🌙 تفسير الحلم
{'='*50}

📊 نوع الحلم: {result['basic_interpretation']['dream_type']}
📈 مستوى الثقة: {result['basic_interpretation']['confidence_level']:.2f}

💭 التفسير الأساسي:
{result['basic_interpretation']['overall_message']}

📋 التوصيات:
"""
                        for i, rec in enumerate(result['recommendations'][:5], 1):
                            interpretation_text += f"{i}. {rec}\n"

                        interpretation_text += f"""

❓ أسئلة المتابعة:
"""
                        for i, q in enumerate(result['follow_up_questions'][:3], 1):
                            interpretation_text += f"{i}. {q}\n"

                        self.dream_result.delete(1.0, tk.END)
                        self.dream_result.insert(1.0, interpretation_text)
                        self.status_text.set("✅ تم تفسير الحلم بنجاح")
                    else:
                        self.dream_result.delete(1.0, tk.END)
                        self.dream_result.insert(1.0, f"❌ خطأ: {result.get('error', 'خطأ غير معروف')}")
                        self.status_text.set("❌ فشل في تفسير الحلم")
                else:
                    messagebox.showerror("خطأ", "نظام تفسير الأحلام غير متاح")

            except Exception as e:
                messagebox.showerror("خطأ", f"خطأ في تفسير الحلم: {str(e)}")
                self.status_text.set("❌ خطأ في التفسير")
            finally:
                self.progress.stop()

        threading.Thread(target=interpret_in_thread, daemon=True).start()

    def execute_code(self):
        """تنفيذ الكود"""
        code = self.code_text.get(1.0, tk.END).strip()
        language = self.code_language.get()

        if not code:
            messagebox.showwarning("تحذير", "يرجى إدخال الكود")
            return

        def execute_in_thread():
            try:
                self.status_text.set("جاري تنفيذ الكود...")
                self.progress.start()

                if self.code_executor:
                    result = self.code_executor.execute(code, ProgrammingLanguage(language))

                    output_text = f"""
💻 نتيجة تنفيذ الكود ({language})
{'='*50}

📤 المخرجات:
{result.stdout}

❌ الأخطاء:
{result.stderr}

📊 معلومات التنفيذ:
• كود الخروج: {result.exit_code}
• وقت التنفيذ: {result.execution_time:.3f} ثانية
"""

                    self.code_output.delete(1.0, tk.END)
                    self.code_output.insert(1.0, output_text)
                    self.status_text.set("✅ تم تنفيذ الكود بنجاح")
                else:
                    messagebox.showerror("خطأ", "منفذ الكود غير متاح")

            except Exception as e:
                messagebox.showerror("خطأ", f"خطأ في تنفيذ الكود: {str(e)}")
                self.status_text.set("❌ خطأ في التنفيذ")
            finally:
                self.progress.stop()

        threading.Thread(target=execute_in_thread, daemon=True).start()

    def generate_image(self):
        """توليد الصورة"""
        prompt = self.image_prompt.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("تحذير", "يرجى إدخال وصف الصورة")
            return

        def generate_in_thread():
            try:
                self.status_text.set("جاري توليد الصورة...")
                self.progress.start()

                if self.image_generator:
                    width = int(self.image_width.get())
                    height = int(self.image_height.get())

                    params = GenerationParameters(
                        mode=GenerationMode.TEXT_TO_IMAGE,
                        width=width,
                        height=height
                    )

                    result = self.image_generator.generate_image(prompt, params)

                    self.image_result.config(text=f"✅ تم توليد الصورة بنجاح\nالوقت: {result.generation_time:.2f} ثانية")
                    self.status_text.set("✅ تم توليد الصورة بنجاح")
                else:
                    messagebox.showerror("خطأ", "مولد الصور غير متاح")

            except Exception as e:
                messagebox.showerror("خطأ", f"خطأ في توليد الصورة: {str(e)}")
                self.status_text.set("❌ خطأ في التوليد")
            finally:
                self.progress.stop()

        threading.Thread(target=generate_in_thread, daemon=True).start()

    def process_arabic(self):
        """معالجة النص العربي"""
        text = self.arabic_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("تحذير", "يرجى إدخال النص العربي")
            return

        def process_in_thread():
            try:
                self.status_text.set("جاري تحليل النص العربي...")
                self.progress.start()

                if self.arabic_processor:
                    result = self.arabic_processor.process_text(text)

                    analysis_text = f"""
📝 تحليل النص العربي
{'='*50}

📄 النص الأصلي:
{text}

🔍 نتيجة التحليل:
{result}

📊 معلومات إضافية:
• عدد الكلمات: {len(text.split())}
• عدد الأحرف: {len(text)}
• تم التحليل في: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

                    self.arabic_result.delete(1.0, tk.END)
                    self.arabic_result.insert(1.0, analysis_text)
                    self.status_text.set("✅ تم تحليل النص بنجاح")
                else:
                    messagebox.showerror("خطأ", "معالج اللغة العربية غير متاح")

            except Exception as e:
                messagebox.showerror("خطأ", f"خطأ في تحليل النص: {str(e)}")
                self.status_text.set("❌ خطأ في التحليل")
            finally:
                self.progress.stop()

        threading.Thread(target=process_in_thread, daemon=True).start()

    def solve_equation(self):
        """حل المعادلة"""
        equation = self.equation_text.get().strip()
        if not equation:
            messagebox.showwarning("تحذير", "يرجى إدخال المعادلة")
            return

        def solve_in_thread():
            try:
                self.status_text.set("جاري حل المعادلة...")
                self.progress.start()

                # إنشاء معادلة عامة
                gse = GeneralShapeEquation(equation_type=EquationType.SHAPE)
                gse.add_component('main', equation)

                # تقييم المعادلة
                sample_values = {'x': 1, 'y': 1, 'z': 1, 'a': 1, 'b': 1, 'c': 1}
                result = gse.evaluate(sample_values)

                solution_text = f"""
🧮 حل المعادلة
{'='*50}

📐 المعادلة:
{equation}

🔢 التقييم عند القيم النموذجية:
{result}

📊 معلومات المعادلة:
• نوع المعادلة: {gse.equation_type.value}
• مستوى التعقيد: {gse.metadata.complexity:.2f}
• معرف المعادلة: {gse.metadata.equation_id}

💡 ملاحظة: هذا تقييم نموذجي للمعادلة
"""

                self.math_result.delete(1.0, tk.END)
                self.math_result.insert(1.0, solution_text)
                self.status_text.set("✅ تم حل المعادلة بنجاح")

            except Exception as e:
                messagebox.showerror("خطأ", f"خطأ في حل المعادلة: {str(e)}")
                self.status_text.set("❌ خطأ في الحل")
            finally:
                self.progress.stop()

        threading.Thread(target=solve_in_thread, daemon=True).start()

    def generate_mind_map(self):
        """إنشاء خريطة ذهنية"""
        topic = self.brainstorm_topic.get().strip()
        if not topic:
            messagebox.showwarning("تحذير", "يرجى إدخال الموضوع الرئيسي")
            return

        # مسح الخريطة السابقة
        self.mind_map_canvas.delete("all")

        # رسم الخريطة الذهنية
        canvas_width = self.mind_map_canvas.winfo_width() or 600
        canvas_height = self.mind_map_canvas.winfo_height() or 400

        center_x = canvas_width // 2
        center_y = canvas_height // 2

        # الموضوع الرئيسي
        self.mind_map_canvas.create_oval(center_x-60, center_y-30, center_x+60, center_y+30,
                                        fill='lightblue', outline='blue', width=2)
        self.mind_map_canvas.create_text(center_x, center_y, text=topic, font=('Arial', 12, 'bold'))

        # الفروع الفرعية
        branches = [
            "الأفكار الرئيسية", "التطبيقات", "التحديات",
            "الحلول", "الفرص", "المخاطر"
        ]

        import math
        for i, branch in enumerate(branches):
            angle = (2 * math.pi * i) / len(branches)
            branch_x = center_x + 150 * math.cos(angle)
            branch_y = center_y + 100 * math.sin(angle)

            # رسم الخط
            self.mind_map_canvas.create_line(center_x, center_y, branch_x, branch_y,
                                           fill='gray', width=2)

            # رسم الفرع
            self.mind_map_canvas.create_oval(branch_x-40, branch_y-20, branch_x+40, branch_y+20,
                                           fill='lightgreen', outline='green')
            self.mind_map_canvas.create_text(branch_x, branch_y, text=branch, font=('Arial', 10))

        self.status_text.set("✅ تم إنشاء الخريطة الذهنية")

    def refresh_system_info(self):
        """تحديث معلومات النظام"""
        info_text = f"""
📊 معلومات نظام بصيرة
{'='*50}

🕒 الوقت الحالي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 حالة المكونات:
• النظام الرئيسي: {'✅ متاح' if self.basira_system else '❌ غير متاح'}
• تفسير الأحلام: {'✅ متاح' if self.dream_system else '❌ غير متاح'}
• تنفيذ الكود: {'✅ متاح' if self.code_executor else '❌ غير متاح'}
• توليد الصور: {'✅ متاح' if self.image_generator else '❌ غير متاح'}
• معالجة العربية: {'✅ متاح' if self.arabic_processor else '❌ غير متاح'}

💾 معلومات النظام:
• إصدار Python: {sys.version.split()[0]}
• المنصة: {sys.platform}
• مجلد العمل: {os.getcwd()}

📈 الإحصائيات:
• عدد التبويبات: {self.notebook.index('end')}
• حالة النظام: {'🟢 جاهز' if BASIRA_AVAILABLE else '🟡 جزئي'}
"""

        self.system_info.delete(1.0, tk.END)
        self.system_info.insert(1.0, info_text)
        self.status_text.set("✅ تم تحديث معلومات النظام")

    # وظائف مساعدة
    def update_time(self):
        """تحديث الوقت"""
        current_time = datetime.now().strftime('%H:%M:%S')
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def show_dream_tab(self):
        """عرض تبويب تفسير الأحلام"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '🌙 تفسير الأحلام':
                self.notebook.select(i)
                break

    def show_code_tab(self):
        """عرض تبويب تنفيذ الكود"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '💻 تنفيذ الكود':
                self.notebook.select(i)
                break

    def show_image_tab(self):
        """عرض تبويب توليد الصور"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '🎨 توليد الصور':
                self.notebook.select(i)
                break

    def show_video_tab(self):
        """عرض تبويب توليد الفيديو"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '🎬 توليد الفيديو':
                self.notebook.select(i)
                break

    def show_arabic_tab(self):
        """عرض تبويب معالجة العربية"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '📝 معالجة العربية':
                self.notebook.select(i)
                break

    def show_math_tab(self):
        """عرض تبويب حل المعادلات"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '🧮 حل المعادلات':
                self.notebook.select(i)
                break

    def show_brainstorm_tab(self):
        """عرض تبويب العصف الذهني"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '🧠 العصف الذهني':
                self.notebook.select(i)
                break

    def show_monitor_tab(self):
        """عرض تبويب مراقبة النظام"""
        for i in range(self.notebook.index('end')):
            if self.notebook.tab(i, 'text') == '📊 مراقبة النظام':
                self.notebook.select(i)
                break

    def show_system_status(self):
        """عرض حالة النظام"""
        status = "✅ جاهز" if BASIRA_AVAILABLE else "⚠️ جزئي"
        messagebox.showinfo("حالة النظام", f"حالة نظام بصيرة: {status}")

    def restart_system(self):
        """إعادة تشغيل النظام"""
        if messagebox.askyesno("إعادة تشغيل", "هل تريد إعادة تشغيل النظام؟"):
            self.initialize_basira_components()

    def show_about(self):
        """عرض معلومات حول النظام"""
        about_text = """
نظام بصيرة - الإصدار 2.0.0

نظام ذكاء اصطناعي متكامل ومبتكر يجمع بين التراث العربي الإسلامي والتقنيات الحديثة

المطورون: فريق تطوير نظام بصيرة
التاريخ: 2025

🌟 "حيث يلتقي التراث بالحداثة" 🌟
"""
        messagebox.showinfo("حول النظام", about_text)

    def show_help(self):
        """عرض المساعدة"""
        help_text = """
دليل استخدام نظام بصيرة:

1. 🌙 تفسير الأحلام: أدخل نص الحلم ومعلومات الرائي
2. 💻 تنفيذ الكود: اكتب الكود واختر اللغة
3. 🎨 توليد الصور: اكتب وصف الصورة المطلوبة
4. 📝 معالجة العربية: أدخل النص العربي للتحليل
5. 🧮 حل المعادلات: أدخل المعادلة الرياضية
6. 🧠 العصف الذهني: أدخل الموضوع لإنشاء خريطة ذهنية

للمساعدة الإضافية، راجع التوثيق الكامل.
"""
        messagebox.showinfo("المساعدة", help_text)

    def run(self):
        """تشغيل التطبيق"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    app = BasiraDesktopApp()
    app.run()


if __name__ == "__main__":
    main()
