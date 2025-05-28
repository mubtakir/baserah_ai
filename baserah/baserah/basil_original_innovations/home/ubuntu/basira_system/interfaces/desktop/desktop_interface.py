#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
واجهة سطح المكتب لنظام بصيرة

هذا الملف يحتوي على تنفيذ واجهة سطح المكتب لنظام بصيرة،
باستخدام PyQt5 لإنشاء واجهة مستخدم رسومية غنية وتفاعلية.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QLineEdit, QTabWidget,
        QSplitter, QTreeView, QListView, QMenu, QAction, QToolBar,
        QStatusBar, QFileDialog, QMessageBox, QDialog, QComboBox,
        QCheckBox, QRadioButton, QGroupBox, QScrollArea, QSizePolicy,
        QFrame, QProgressBar, QSystemTrayIcon
    )
    from PyQt5.QtGui import (
        QIcon, QPixmap, QFont, QColor, QPalette, QTextCursor,
        QStandardItemModel, QStandardItem, QKeySequence, QTextDocument
    )
    from PyQt5.QtCore import (
        Qt, QSize, QThread, pyqtSignal, pyqtSlot, QTimer,
        QModelIndex, QPoint, QRect, QUrl, QSettings
    )
except ImportError:
    logging.error("PyQt5 is not installed. Please install it using: pip install PyQt5")
    sys.exit(1)

# إضافة المسار إلى حزم النظام
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer
from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer
from code_execution.executor import CodeExecutionService, ProgrammingLanguage, ExecutionMode

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('basira_system.interfaces.desktop')


class ThemeType(Enum):
    """أنواع السمات."""
    LIGHT = auto()  # سمة فاتحة
    DARK = auto()  # سمة داكنة
    SYSTEM = auto()  # سمة النظام


class ViewMode(Enum):
    """أنماط العرض."""
    SIMPLE = auto()  # عرض بسيط
    ADVANCED = auto()  # عرض متقدم
    EXPERT = auto()  # عرض خبير


class BasiraDesktopApp(QMainWindow):
    """تطبيق سطح المكتب لنظام بصيرة."""
    
    def __init__(self):
        """تهيئة التطبيق."""
        super().__init__()
        
        self.logger = logging.getLogger('basira_system.interfaces.desktop.app')
        
        # تهيئة الإعدادات
        self.settings = QSettings("BasiraSystem", "DesktopApp")
        
        # تهيئة المكونات
        self.root_extractor = ArabicRootExtractor()
        self.syntax_analyzer = ArabicSyntaxAnalyzer()
        self.rhetoric_analyzer = ArabicRhetoricAnalyzer()
        self.code_execution_service = CodeExecutionService()
        
        # تهيئة واجهة المستخدم
        self.init_ui()
        
        # تطبيق الإعدادات المحفوظة
        self.load_settings()
    
    def init_ui(self):
        """تهيئة واجهة المستخدم."""
        # إعداد النافذة الرئيسية
        self.setWindowTitle("نظام بصيرة - واجهة سطح المكتب")
        self.setMinimumSize(1024, 768)
        
        # إنشاء القائمة الرئيسية
        self.create_menu()
        
        # إنشاء شريط الأدوات
        self.create_toolbar()
        
        # إنشاء الحالة السفلية
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("جاهز")
        
        # إنشاء الويدجت المركزي
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # إنشاء التخطيط الرئيسي
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # إنشاء مقسم رئيسي
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # إنشاء لوحة التنقل
        self.create_navigation_panel()
        
        # إنشاء لوحة المحتوى
        self.create_content_panel()
        
        # ضبط نسب المقسم
        self.main_splitter.setSizes([200, 800])
        
        # إنشاء شريط التقدم
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
    
    def create_menu(self):
        """إنشاء القائمة الرئيسية."""
        # قائمة الملف
        file_menu = self.menuBar().addMenu("ملف")
        
        # إجراءات قائمة الملف
        new_action = QAction(QIcon.fromTheme("document-new"), "جديد", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)
        
        open_action = QAction(QIcon.fromTheme("document-open"), "فتح", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction(QIcon.fromTheme("document-save"), "حفظ", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction(QIcon.fromTheme("document-save-as"), "حفظ باسم", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction(QIcon.fromTheme("application-exit"), "خروج", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # قائمة تحرير
        edit_menu = self.menuBar().addMenu("تحرير")
        
        # إجراءات قائمة تحرير
        undo_action = QAction(QIcon.fromTheme("edit-undo"), "تراجع", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction(QIcon.fromTheme("edit-redo"), "إعادة", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction(QIcon.fromTheme("edit-cut"), "قص", self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(self.cut)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction(QIcon.fromTheme("edit-copy"), "نسخ", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction(QIcon.fromTheme("edit-paste"), "لصق", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.paste)
        edit_menu.addAction(paste_action)
        
        # قائمة عرض
        view_menu = self.menuBar().addMenu("عرض")
        
        # إجراءات قائمة عرض
        theme_menu = view_menu.addMenu("السمة")
        
        light_theme_action = QAction("فاتحة", self)
        light_theme_action.setCheckable(True)
        light_theme_action.triggered.connect(lambda: self.set_theme(ThemeType.LIGHT))
        theme_menu.addAction(light_theme_action)
        
        dark_theme_action = QAction("داكنة", self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.triggered.connect(lambda: self.set_theme(ThemeType.DARK))
        theme_menu.addAction(dark_theme_action)
        
        system_theme_action = QAction("النظام", self)
        system_theme_action.setCheckable(True)
        system_theme_action.triggered.connect(lambda: self.set_theme(ThemeType.SYSTEM))
        theme_menu.addAction(system_theme_action)
        
        view_menu.addSeparator()
        
        view_mode_menu = view_menu.addMenu("نمط العرض")
        
        simple_mode_action = QAction("بسيط", self)
        simple_mode_action.setCheckable(True)
        simple_mode_action.triggered.connect(lambda: self.set_view_mode(ViewMode.SIMPLE))
        view_mode_menu.addAction(simple_mode_action)
        
        advanced_mode_action = QAction("متقدم", self)
        advanced_mode_action.setCheckable(True)
        advanced_mode_action.triggered.connect(lambda: self.set_view_mode(ViewMode.ADVANCED))
        view_mode_menu.addAction(advanced_mode_action)
        
        expert_mode_action = QAction("خبير", self)
        expert_mode_action.setCheckable(True)
        expert_mode_action.triggered.connect(lambda: self.set_view_mode(ViewMode.EXPERT))
        view_mode_menu.addAction(expert_mode_action)
        
        # قائمة أدوات
        tools_menu = self.menuBar().addMenu("أدوات")
        
        # إجراءات قائمة أدوات
        nlp_menu = tools_menu.addMenu("معالجة اللغة العربية")
        
        root_extraction_action = QAction("استخراج الجذور", self)
        root_extraction_action.triggered.connect(self.show_root_extraction_tool)
        nlp_menu.addAction(root_extraction_action)
        
        syntax_analysis_action = QAction("تحليل نحوي", self)
        syntax_analysis_action.triggered.connect(self.show_syntax_analysis_tool)
        nlp_menu.addAction(syntax_analysis_action)
        
        rhetoric_analysis_action = QAction("تحليل بلاغي", self)
        rhetoric_analysis_action.triggered.connect(self.show_rhetoric_analysis_tool)
        nlp_menu.addAction(rhetoric_analysis_action)
        
        tools_menu.addSeparator()
        
        code_execution_action = QAction("تنفيذ الأكواد", self)
        code_execution_action.triggered.connect(self.show_code_execution_tool)
        tools_menu.addAction(code_execution_action)
        
        # قائمة مساعدة
        help_menu = self.menuBar().addMenu("مساعدة")
        
        # إجراءات قائمة مساعدة
        about_action = QAction("حول", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        help_action = QAction("مساعدة", self)
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def create_toolbar(self):
        """إنشاء شريط الأدوات."""
        # شريط الأدوات الرئيسي
        self.main_toolbar = QToolBar("شريط الأدوات الرئيسي")
        self.addToolBar(self.main_toolbar)
        
        # إضافة إجراءات شريط الأدوات
        new_action = QAction(QIcon.fromTheme("document-new"), "جديد", self)
        new_action.triggered.connect(self.new_file)
        self.main_toolbar.addAction(new_action)
        
        open_action = QAction(QIcon.fromTheme("document-open"), "فتح", self)
        open_action.triggered.connect(self.open_file)
        self.main_toolbar.addAction(open_action)
        
        save_action = QAction(QIcon.fromTheme("document-save"), "حفظ", self)
        save_action.triggered.connect(self.save_file)
        self.main_toolbar.addAction(save_action)
        
        self.main_toolbar.addSeparator()
        
        cut_action = QAction(QIcon.fromTheme("edit-cut"), "قص", self)
        cut_action.triggered.connect(self.cut)
        self.main_toolbar.addAction(cut_action)
        
        copy_action = QAction(QIcon.fromTheme("edit-copy"), "نسخ", self)
        copy_action.triggered.connect(self.copy)
        self.main_toolbar.addAction(copy_action)
        
        paste_action = QAction(QIcon.fromTheme("edit-paste"), "لصق", self)
        paste_action.triggered.connect(self.paste)
        self.main_toolbar.addAction(paste_action)
        
        self.main_toolbar.addSeparator()
        
        run_action = QAction(QIcon.fromTheme("media-playback-start"), "تشغيل", self)
        run_action.triggered.connect(self.run_current_code)
        self.main_toolbar.addAction(run_action)
    
    def create_navigation_panel(self):
        """إنشاء لوحة التنقل."""
        # إنشاء ويدجت لوحة التنقل
        self.navigation_panel = QWidget()
        self.navigation_layout = QVBoxLayout(self.navigation_panel)
        
        # إنشاء عنوان اللوحة
        self.navigation_title = QLabel("التنقل")
        self.navigation_title.setAlignment(Qt.AlignCenter)
        self.navigation_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.navigation_layout.addWidget(self.navigation_title)
        
        # إنشاء شجرة التنقل
        self.navigation_tree = QTreeView()
        self.navigation_tree.setHeaderHidden(True)
        self.navigation_layout.addWidget(self.navigation_tree)
        
        # إنشاء نموذج البيانات للشجرة
        self.navigation_model = QStandardItemModel()
        self.navigation_tree.setModel(self.navigation_model)
        
        # إضافة العناصر الرئيسية
        self.add_navigation_items()
        
        # إضافة لوحة التنقل إلى المقسم الرئيسي
        self.main_splitter.addWidget(self.navigation_panel)
    
    def add_navigation_items(self):
        """إضافة عناصر التنقل."""
        # إضافة العناصر الرئيسية
        root_item = self.navigation_model.invisibleRootItem()
        
        # عنصر الرئيسية
        home_item = QStandardItem("الرئيسية")
        home_item.setData("home")
        root_item.appendRow(home_item)
        
        # عنصر معالجة اللغة العربية
        nlp_item = QStandardItem("معالجة اللغة العربية")
        nlp_item.setData("nlp")
        
        # إضافة العناصر الفرعية لمعالجة اللغة العربية
        morphology_item = QStandardItem("الصرف")
        morphology_item.setData("morphology")
        nlp_item.appendRow(morphology_item)
        
        syntax_item = QStandardItem("النحو")
        syntax_item.setData("syntax")
        nlp_item.appendRow(syntax_item)
        
        rhetoric_item = QStandardItem("البلاغة")
        rhetoric_item.setData("rhetoric")
        nlp_item.appendRow(rhetoric_item)
        
        root_item.appendRow(nlp_item)
        
        # عنصر تنفيذ الأكواد
        code_execution_item = QStandardItem("تنفيذ الأكواد")
        code_execution_item.setData("code_execution")
        root_item.appendRow(code_execution_item)
        
        # عنصر توليد المحتوى
        content_generation_item = QStandardItem("توليد المحتوى")
        content_generation_item.setData("content_generation")
        
        # إضافة العناصر الفرعية لتوليد المحتوى
        text_generation_item = QStandardItem("توليد النصوص")
        text_generation_item.setData("text_generation")
        content_generation_item.appendRow(text_generation_item)
        
        image_generation_item = QStandardItem("توليد الصور")
        image_generation_item.setData("image_generation")
        content_generation_item.appendRow(image_generation_item)
        
        video_generation_item = QStandardItem("توليد الفيديو")
        video_generation_item.setData("video_generation")
        content_generation_item.appendRow(video_generation_item)
        
        root_item.appendRow(content_generation_item)
        
        # عنصر تصور المعرفة
        knowledge_visualization_item = QStandardItem("تصور المعرفة")
        knowledge_visualization_item.setData("knowledge_visualization")
        root_item.appendRow(knowledge_visualization_item)
        
        # عنصر الإعدادات
        settings_item = QStandardItem("الإعدادات")
        settings_item.setData("settings")
        root_item.appendRow(settings_item)
        
        # ربط حدث النقر على العناصر
        self.navigation_tree.clicked.connect(self.on_navigation_item_clicked)
    
    def create_content_panel(self):
        """إنشاء لوحة المحتوى."""
        # إنشاء ويدجت لوحة المحتوى
        self.content_panel = QWidget()
        self.content_layout = QVBoxLayout(self.content_panel)
        
        # إنشاء علامات التبويب
        self.content_tabs = QTabWidget()
        self.content_tabs.setTabsClosable(True)
        self.content_tabs.tabCloseRequested.connect(self.close_tab)
        self.content_layout.addWidget(self.content_tabs)
        
        # إضافة علامة تبويب الترحيب
        self.add_welcome_tab()
        
        # إضافة لوحة المحتوى إلى المقسم الرئيسي
        self.main_splitter.addWidget(self.content_panel)
    
    def add_welcome_tab(self):
        """إضافة علامة تبويب الترحيب."""
        # إنشاء ويدجت الترحيب
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        
        # إضافة شعار النظام
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png")  # يجب توفير ملف الشعار
        if logo_pixmap.isNull():
            # إذا لم يتم العثور على ملف الشعار، استخدم نصًا بديلًا
            logo_label.setText("نظام بصيرة")
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #2980b9;")
        else:
            logo_label.setPixmap(logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignCenter)
        
        welcome_layout.addWidget(logo_label)
        
        # إضافة نص الترحيب
        welcome_text = QLabel("مرحبًا بك في نظام بصيرة")
        welcome_text.setAlignment(Qt.AlignCenter)
        welcome_text.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 20px;")
        welcome_layout.addWidget(welcome_text)
        
        # إضافة وصف النظام
        description_text = QLabel(
            "نظام بصيرة هو نموذج لغوي معرفي توليدي ونظام ذكاء اصطناعي مبتكر "
            "يجمع بين المعالجة الرمزية والتعلم العميق والمعزز، "
            "مع قدرات متقدمة في معالجة اللغة العربية وتوليد المحتوى."
        )
        description_text.setAlignment(Qt.AlignCenter)
        description_text.setWordWrap(True)
        description_text.setStyleSheet("font-size: 14px; margin: 20px;")
        welcome_layout.addWidget(description_text)
        
        # إضافة أزرار الإجراءات السريعة
        quick_actions_layout = QHBoxLayout()
        
        # زر معالجة اللغة العربية
        nlp_button = QPushButton("معالجة اللغة العربية")
        nlp_button.clicked.connect(lambda: self.on_navigation_item_clicked(self.navigation_model.indexFromItem(self.navigation_model.findItems("معالجة اللغة العربية")[0])))
        quick_actions_layout.addWidget(nlp_button)
        
        # زر تنفيذ الأكواد
        code_execution_button = QPushButton("تنفيذ الأكواد")
        code_execution_button.clicked.connect(lambda: self.on_navigation_item_clicked(self.navigation_model.indexFromItem(self.navigation_model.findItems("تنفيذ الأكواد")[0])))
        quick_actions_layout.addWidget(code_execution_button)
        
        # زر توليد المحتوى
        content_generation_button = QPushButton("توليد المحتوى")
        content_generation_button.clicked.connect(lambda: self.on_navigation_item_clicked(self.navigation_model.indexFromItem(self.navigation_model.findItems("توليد المحتوى")[0])))
        quick_actions_layout.addWidget(content_generation_button)
        
        welcome_layout.addLayout(quick_actions_layout)
        
        # إضافة مساحة فارغة
        welcome_layout.addStretch()
        
        # إضافة معلومات الإصدار
        version_text = QLabel("الإصدار: 1.0.0")
        version_text.setAlignment(Qt.AlignCenter)
        version_text.setStyleSheet("font-size: 12px; color: gray;")
        welcome_layout.addWidget(version_text)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(welcome_widget, "الترحيب")
    
    def on_navigation_item_clicked(self, index):
        """
        معالجة النقر على عنصر التنقل.
        
        Args:
            index: مؤشر العنصر المنقور
        """
        # الحصول على بيانات العنصر
        item = self.navigation_model.itemFromIndex(index)
        item_data = item.data()
        
        # معالجة النقر حسب نوع العنصر
        if item_data == "home":
            # عرض الصفحة الرئيسية
            self.show_home_page()
        elif item_data == "morphology":
            # عرض أداة الصرف
            self.show_morphology_tool()
        elif item_data == "syntax":
            # عرض أداة النحو
            self.show_syntax_tool()
        elif item_data == "rhetoric":
            # عرض أداة البلاغة
            self.show_rhetoric_tool()
        elif item_data == "code_execution":
            # عرض أداة تنفيذ الأكواد
            self.show_code_execution_tool()
        elif item_data == "text_generation":
            # عرض أداة توليد النصوص
            self.show_text_generation_tool()
        elif item_data == "image_generation":
            # عرض أداة توليد الصور
            self.show_image_generation_tool()
        elif item_data == "video_generation":
            # عرض أداة توليد الفيديو
            self.show_video_generation_tool()
        elif item_data == "knowledge_visualization":
            # عرض أداة تصور المعرفة
            self.show_knowledge_visualization_tool()
        elif item_data == "settings":
            # عرض الإعدادات
            self.show_settings()
    
    def show_home_page(self):
        """عرض الصفحة الرئيسية."""
        # البحث عن علامة تبويب الترحيب
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "الترحيب":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إضافة علامة تبويب الترحيب إذا لم تكن موجودة
        self.add_welcome_tab()
    
    def show_morphology_tool(self):
        """عرض أداة الصرف."""
        # البحث عن علامة تبويب الصرف
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "الصرف":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت الصرف
        morphology_widget = QWidget()
        morphology_layout = QVBoxLayout(morphology_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة الصرف")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        morphology_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل نصًا عربيًا لاستخراج جذور الكلمات وتحليلها صرفيًا.")
        description_label.setWordWrap(True)
        morphology_layout.addWidget(description_label)
        
        # إضافة حقل إدخال النص
        input_layout = QHBoxLayout()
        input_label = QLabel("النص:")
        input_layout.addWidget(input_label)
        
        self.morphology_input = QTextEdit()
        self.morphology_input.setPlaceholderText("أدخل النص هنا...")
        input_layout.addWidget(self.morphology_input)
        
        morphology_layout.addLayout(input_layout)
        
        # إضافة زر التحليل
        analyze_button = QPushButton("تحليل")
        analyze_button.clicked.connect(self.analyze_morphology)
        morphology_layout.addWidget(analyze_button)
        
        # إضافة حقل النتائج
        results_label = QLabel("النتائج:")
        morphology_layout.addWidget(results_label)
        
        self.morphology_results = QTextEdit()
        self.morphology_results.setReadOnly(True)
        morphology_layout.addWidget(self.morphology_results)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(morphology_widget, "الصرف")
        self.content_tabs.setCurrentWidget(morphology_widget)
    
    def analyze_morphology(self):
        """تحليل النص صرفيًا."""
        # الحصول على النص المدخل
        text = self.morphology_input.toPlainText()
        
        if not text:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال نص للتحليل.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # تحليل النص
            self.statusBar.showMessage("جاري التحليل...")
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(50)
            
            # استخراج الجذور
            results = []
            words = text.split()
            
            for word in words:
                root = self.root_extractor.extract_root(word)
                results.append(f"الكلمة: {word} - الجذر: {root}")
            
            # عرض النتائج
            self.morphology_results.setPlainText("\n".join(results))
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم التحليل بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error analyzing morphology: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء التحليل: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التحليل.")
    
    def show_syntax_tool(self):
        """عرض أداة النحو."""
        # البحث عن علامة تبويب النحو
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "النحو":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت النحو
        syntax_widget = QWidget()
        syntax_layout = QVBoxLayout(syntax_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة النحو")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        syntax_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل نصًا عربيًا لتحليله نحويًا وتحديد مكونات الجملة.")
        description_label.setWordWrap(True)
        syntax_layout.addWidget(description_label)
        
        # إضافة حقل إدخال النص
        input_layout = QHBoxLayout()
        input_label = QLabel("النص:")
        input_layout.addWidget(input_label)
        
        self.syntax_input = QTextEdit()
        self.syntax_input.setPlaceholderText("أدخل النص هنا...")
        input_layout.addWidget(self.syntax_input)
        
        syntax_layout.addLayout(input_layout)
        
        # إضافة زر التحليل
        analyze_button = QPushButton("تحليل")
        analyze_button.clicked.connect(self.analyze_syntax)
        syntax_layout.addWidget(analyze_button)
        
        # إضافة حقل النتائج
        results_label = QLabel("النتائج:")
        syntax_layout.addWidget(results_label)
        
        self.syntax_results = QTextEdit()
        self.syntax_results.setReadOnly(True)
        syntax_layout.addWidget(self.syntax_results)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(syntax_widget, "النحو")
        self.content_tabs.setCurrentWidget(syntax_widget)
    
    def analyze_syntax(self):
        """تحليل النص نحويًا."""
        # الحصول على النص المدخل
        text = self.syntax_input.toPlainText()
        
        if not text:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال نص للتحليل.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # تحليل النص
            self.statusBar.showMessage("جاري التحليل...")
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(50)
            
            # تحليل النحو
            sentences = self.syntax_analyzer.analyze(text)
            
            # عرض النتائج
            results = []
            
            for i, sentence in enumerate(sentences):
                results.append(f"الجملة {i+1}: {sentence.text}")
                results.append("المكونات:")
                
                for j, token in enumerate(sentence.tokens):
                    results.append(f"  {j+1}. {token.text} - {token.pos_tag.name}")
                
                results.append("")
            
            self.syntax_results.setPlainText("\n".join(results))
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم التحليل بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error analyzing syntax: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء التحليل: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التحليل.")
    
    def show_rhetoric_tool(self):
        """عرض أداة البلاغة."""
        # البحث عن علامة تبويب البلاغة
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "البلاغة":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت البلاغة
        rhetoric_widget = QWidget()
        rhetoric_layout = QVBoxLayout(rhetoric_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة البلاغة")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        rhetoric_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل نصًا عربيًا لتحليله بلاغيًا وتحديد الأساليب البلاغية.")
        description_label.setWordWrap(True)
        rhetoric_layout.addWidget(description_label)
        
        # إضافة حقل إدخال النص
        input_layout = QHBoxLayout()
        input_label = QLabel("النص:")
        input_layout.addWidget(input_label)
        
        self.rhetoric_input = QTextEdit()
        self.rhetoric_input.setPlaceholderText("أدخل النص هنا...")
        input_layout.addWidget(self.rhetoric_input)
        
        rhetoric_layout.addLayout(input_layout)
        
        # إضافة زر التحليل
        analyze_button = QPushButton("تحليل")
        analyze_button.clicked.connect(self.analyze_rhetoric)
        rhetoric_layout.addWidget(analyze_button)
        
        # إضافة حقل النتائج
        results_label = QLabel("النتائج:")
        rhetoric_layout.addWidget(results_label)
        
        self.rhetoric_results = QTextEdit()
        self.rhetoric_results.setReadOnly(True)
        rhetoric_layout.addWidget(self.rhetoric_results)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(rhetoric_widget, "البلاغة")
        self.content_tabs.setCurrentWidget(rhetoric_widget)
    
    def analyze_rhetoric(self):
        """تحليل النص بلاغيًا."""
        # الحصول على النص المدخل
        text = self.rhetoric_input.toPlainText()
        
        if not text:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال نص للتحليل.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # تحليل النص
            self.statusBar.showMessage("جاري التحليل...")
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(50)
            
            # تحليل البلاغة
            rhetoric_elements = self.rhetoric_analyzer.analyze(text)
            
            # عرض النتائج
            results = []
            
            if rhetoric_elements:
                results.append("الأساليب البلاغية:")
                
                for i, element in enumerate(rhetoric_elements):
                    results.append(f"{i+1}. {element.rhetoric_type.name}: {element.text}")
                    results.append(f"   الفئة: {element.category.name}")
                    results.append(f"   الشرح: {element.explanation}")
                    results.append(f"   مستوى الثقة: {element.confidence:.2f}")
                    results.append("")
            else:
                results.append("لم يتم العثور على أساليب بلاغية في النص.")
            
            self.rhetoric_results.setPlainText("\n".join(results))
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم التحليل بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error analyzing rhetoric: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء التحليل: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التحليل.")
    
    def show_code_execution_tool(self):
        """عرض أداة تنفيذ الأكواد."""
        # البحث عن علامة تبويب تنفيذ الأكواد
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "تنفيذ الأكواد":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت تنفيذ الأكواد
        code_execution_widget = QWidget()
        code_execution_layout = QVBoxLayout(code_execution_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة تنفيذ الأكواد")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        code_execution_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل كودًا برمجيًا لتنفيذه وعرض النتائج.")
        description_label.setWordWrap(True)
        code_execution_layout.addWidget(description_label)
        
        # إضافة اختيار لغة البرمجة
        language_layout = QHBoxLayout()
        language_label = QLabel("لغة البرمجة:")
        language_layout.addWidget(language_label)
        
        self.language_combo = QComboBox()
        self.language_combo.addItem("Python", ProgrammingLanguage.PYTHON)
        self.language_combo.addItem("JavaScript", ProgrammingLanguage.JAVASCRIPT)
        self.language_combo.addItem("TypeScript", ProgrammingLanguage.TYPESCRIPT)
        self.language_combo.addItem("Bash", ProgrammingLanguage.BASH)
        language_layout.addWidget(self.language_combo)
        
        code_execution_layout.addLayout(language_layout)
        
        # إضافة حقل إدخال الكود
        code_label = QLabel("الكود:")
        code_execution_layout.addWidget(code_label)
        
        self.code_input = QTextEdit()
        self.code_input.setPlaceholderText("أدخل الكود هنا...")
        self.code_input.setStyleSheet("font-family: monospace;")
        code_execution_layout.addWidget(self.code_input)
        
        # إضافة حقل إدخال البيانات
        input_layout = QHBoxLayout()
        input_label = QLabel("بيانات الإدخال (اختياري):")
        input_layout.addWidget(input_label)
        
        self.input_data = QTextEdit()
        self.input_data.setPlaceholderText("أدخل بيانات الإدخال هنا...")
        self.input_data.setMaximumHeight(100)
        input_layout.addWidget(self.input_data)
        
        code_execution_layout.addLayout(input_layout)
        
        # إضافة زر التنفيذ
        execute_button = QPushButton("تنفيذ")
        execute_button.clicked.connect(self.execute_code)
        code_execution_layout.addWidget(execute_button)
        
        # إضافة حقل النتائج
        results_label = QLabel("النتائج:")
        code_execution_layout.addWidget(results_label)
        
        self.code_results = QTextEdit()
        self.code_results.setReadOnly(True)
        self.code_results.setStyleSheet("font-family: monospace;")
        code_execution_layout.addWidget(self.code_results)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(code_execution_widget, "تنفيذ الأكواد")
        self.content_tabs.setCurrentWidget(code_execution_widget)
    
    def execute_code(self):
        """تنفيذ الكود."""
        # الحصول على الكود المدخل
        code = self.code_input.toPlainText()
        
        if not code:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال كود للتنفيذ.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # تنفيذ الكود
            self.statusBar.showMessage("جاري التنفيذ...")
            
            # الحصول على لغة البرمجة
            language = self.language_combo.currentData()
            
            # الحصول على بيانات الإدخال
            input_data = self.input_data.toPlainText()
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(50)
            
            # تنفيذ الكود
            result = self.code_execution_service.execute(
                code,
                language=language,
                input_data=input_data,
                timeout=30
            )
            
            # عرض النتائج
            output = []
            
            output.append(f"الحالة: {result.status.name}")
            output.append(f"وقت التنفيذ: {result.execution_time:.2f} ثانية")
            
            if result.output:
                output.append("\nالمخرجات:")
                output.append(result.output)
            
            if result.error:
                output.append("\nالأخطاء:")
                output.append(result.error)
            
            self.code_results.setPlainText("\n".join(output))
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم التنفيذ بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error executing code: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء التنفيذ: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التنفيذ.")
    
    def show_text_generation_tool(self):
        """عرض أداة توليد النصوص."""
        # البحث عن علامة تبويب توليد النصوص
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "توليد النصوص":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت توليد النصوص
        text_generation_widget = QWidget()
        text_generation_layout = QVBoxLayout(text_generation_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة توليد النصوص")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        text_generation_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل وصفًا أو موضوعًا لتوليد نص حوله.")
        description_label.setWordWrap(True)
        text_generation_layout.addWidget(description_label)
        
        # إضافة حقل إدخال الوصف
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("الوصف:")
        prompt_layout.addWidget(prompt_label)
        
        self.text_prompt = QTextEdit()
        self.text_prompt.setPlaceholderText("أدخل وصفًا أو موضوعًا هنا...")
        prompt_layout.addWidget(self.text_prompt)
        
        text_generation_layout.addLayout(prompt_layout)
        
        # إضافة خيارات التوليد
        options_group = QGroupBox("خيارات التوليد")
        options_layout = QVBoxLayout(options_group)
        
        # طول النص
        length_layout = QHBoxLayout()
        length_label = QLabel("طول النص:")
        length_layout.addWidget(length_label)
        
        self.length_combo = QComboBox()
        self.length_combo.addItem("قصير", "short")
        self.length_combo.addItem("متوسط", "medium")
        self.length_combo.addItem("طويل", "long")
        length_layout.addWidget(self.length_combo)
        
        options_layout.addLayout(length_layout)
        
        # نوع النص
        type_layout = QHBoxLayout()
        type_label = QLabel("نوع النص:")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItem("عام", "general")
        self.type_combo.addItem("أكاديمي", "academic")
        self.type_combo.addItem("أدبي", "literary")
        self.type_combo.addItem("تقني", "technical")
        type_layout.addWidget(self.type_combo)
        
        options_layout.addLayout(type_layout)
        
        text_generation_layout.addWidget(options_group)
        
        # إضافة زر التوليد
        generate_button = QPushButton("توليد")
        generate_button.clicked.connect(self.generate_text)
        text_generation_layout.addWidget(generate_button)
        
        # إضافة حقل النتائج
        results_label = QLabel("النص المولد:")
        text_generation_layout.addWidget(results_label)
        
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        text_generation_layout.addWidget(self.text_results)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(text_generation_widget, "توليد النصوص")
        self.content_tabs.setCurrentWidget(text_generation_widget)
    
    def generate_text(self):
        """توليد النص."""
        # الحصول على الوصف المدخل
        prompt = self.text_prompt.toPlainText()
        
        if not prompt:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال وصف أو موضوع لتوليد النص.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # توليد النص
            self.statusBar.showMessage("جاري التوليد...")
            
            # الحصول على خيارات التوليد
            length = self.length_combo.currentData()
            text_type = self.type_combo.currentData()
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(30)
            
            # محاكاة عملية التوليد (يجب استبدالها بالتنفيذ الفعلي)
            QTimer.singleShot(1000, lambda: self.progress_bar.setValue(60))
            
            # محاكاة النص المولد (يجب استبدالها بالتنفيذ الفعلي)
            generated_text = f"هذا نص تجريبي مولد بناءً على الوصف: '{prompt}'\n\n"
            
            if length == "short":
                generated_text += "هذا نص قصير يتناول الموضوع المطلوب بشكل مختصر ومفيد."
            elif length == "medium":
                generated_text += "هذا نص متوسط الطول يتناول الموضوع المطلوب بشكل متوازن، مع تقديم معلومات كافية وأمثلة توضيحية. يحتوي النص على عدة فقرات تغطي جوانب مختلفة من الموضوع."
            else:  # long
                generated_text += "هذا نص طويل ومفصل يتناول الموضوع المطلوب بعمق وشمولية. يحتوي النص على مقدمة تعريفية، وعدة أقسام رئيسية، وخاتمة تلخص النقاط الأساسية. كما يتضمن النص أمثلة توضيحية، واقتباسات، وإحصائيات، ومراجع لمصادر موثوقة.\n\n"
                generated_text += "القسم الأول يتناول الخلفية التاريخية للموضوع، والتطورات الرئيسية التي مر بها عبر الزمن. القسم الثاني يستعرض المفاهيم الأساسية والنظريات المرتبطة بالموضوع. القسم الثالث يناقش التطبيقات العملية والأمثلة الواقعية. القسم الرابع يتطرق إلى التحديات والفرص المستقبلية.\n\n"
                generated_text += "في الختام، يمكن القول إن هذا الموضوع يمثل أهمية كبيرة في مجاله، ويستحق المزيد من الدراسة والبحث."
            
            # عرض النص المولد
            self.text_results.setPlainText(generated_text)
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم التوليد بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء توليد النص: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التوليد.")
    
    def show_image_generation_tool(self):
        """عرض أداة توليد الصور."""
        # البحث عن علامة تبويب توليد الصور
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "توليد الصور":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت توليد الصور
        image_generation_widget = QWidget()
        image_generation_layout = QVBoxLayout(image_generation_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة توليد الصور")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        image_generation_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل وصفًا لتوليد صورة بناءً عليه.")
        description_label.setWordWrap(True)
        image_generation_layout.addWidget(description_label)
        
        # إضافة حقل إدخال الوصف
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("الوصف:")
        prompt_layout.addWidget(prompt_label)
        
        self.image_prompt = QTextEdit()
        self.image_prompt.setPlaceholderText("أدخل وصفًا للصورة هنا...")
        prompt_layout.addWidget(self.image_prompt)
        
        image_generation_layout.addLayout(prompt_layout)
        
        # إضافة خيارات التوليد
        options_group = QGroupBox("خيارات التوليد")
        options_layout = QVBoxLayout(options_group)
        
        # حجم الصورة
        size_layout = QHBoxLayout()
        size_label = QLabel("حجم الصورة:")
        size_layout.addWidget(size_label)
        
        self.size_combo = QComboBox()
        self.size_combo.addItem("256x256", "256x256")
        self.size_combo.addItem("512x512", "512x512")
        self.size_combo.addItem("1024x1024", "1024x1024")
        size_layout.addWidget(self.size_combo)
        
        options_layout.addLayout(size_layout)
        
        # أسلوب الصورة
        style_layout = QHBoxLayout()
        style_label = QLabel("أسلوب الصورة:")
        style_layout.addWidget(style_label)
        
        self.style_combo = QComboBox()
        self.style_combo.addItem("واقعي", "realistic")
        self.style_combo.addItem("فني", "artistic")
        self.style_combo.addItem("كرتوني", "cartoon")
        self.style_combo.addItem("مجرد", "abstract")
        style_layout.addWidget(self.style_combo)
        
        options_layout.addLayout(style_layout)
        
        image_generation_layout.addWidget(options_group)
        
        # إضافة زر التوليد
        generate_button = QPushButton("توليد")
        generate_button.clicked.connect(self.generate_image)
        image_generation_layout.addWidget(generate_button)
        
        # إضافة حقل عرض الصورة
        results_label = QLabel("الصورة المولدة:")
        image_generation_layout.addWidget(results_label)
        
        self.image_result = QLabel()
        self.image_result.setAlignment(Qt.AlignCenter)
        self.image_result.setMinimumHeight(300)
        self.image_result.setStyleSheet("border: 1px solid gray;")
        image_generation_layout.addWidget(self.image_result)
        
        # إضافة زر حفظ الصورة
        save_button = QPushButton("حفظ الصورة")
        save_button.clicked.connect(self.save_generated_image)
        image_generation_layout.addWidget(save_button)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(image_generation_widget, "توليد الصور")
        self.content_tabs.setCurrentWidget(image_generation_widget)
    
    def generate_image(self):
        """توليد الصورة."""
        # الحصول على الوصف المدخل
        prompt = self.image_prompt.toPlainText()
        
        if not prompt:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال وصف للصورة.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # توليد الصورة
            self.statusBar.showMessage("جاري التوليد...")
            
            # الحصول على خيارات التوليد
            size = self.size_combo.currentData()
            style = self.style_combo.currentData()
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(30)
            
            # محاكاة عملية التوليد (يجب استبدالها بالتنفيذ الفعلي)
            QTimer.singleShot(1000, lambda: self.progress_bar.setValue(60))
            
            # محاكاة الصورة المولدة (يجب استبدالها بالتنفيذ الفعلي)
            # هنا نستخدم صورة تجريبية بدلاً من الصورة المولدة
            placeholder_image = QPixmap(512, 512)
            placeholder_image.fill(QColor(200, 200, 200))
            
            # عرض الصورة المولدة
            self.image_result.setPixmap(placeholder_image)
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم التوليد بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء توليد الصورة: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التوليد.")
    
    def save_generated_image(self):
        """حفظ الصورة المولدة."""
        # التحقق من وجود صورة
        if self.image_result.pixmap() is None or self.image_result.pixmap().isNull():
            QMessageBox.warning(self, "تنبيه", "لا توجد صورة لحفظها.")
            return
        
        try:
            # فتح مربع حوار حفظ الملف
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "حفظ الصورة",
                os.path.expanduser("~/generated_image.png"),
                "صور (*.png *.jpg *.jpeg)"
            )
            
            if file_path:
                # حفظ الصورة
                self.image_result.pixmap().save(file_path)
                self.statusBar.showMessage(f"تم حفظ الصورة في: {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving image: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء حفظ الصورة: {e}")
    
    def show_video_generation_tool(self):
        """عرض أداة توليد الفيديو."""
        # البحث عن علامة تبويب توليد الفيديو
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "توليد الفيديو":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت توليد الفيديو
        video_generation_widget = QWidget()
        video_generation_layout = QVBoxLayout(video_generation_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة توليد الفيديو")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        video_generation_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("أدخل وصفًا لتوليد مقطع فيديو بناءً عليه.")
        description_label.setWordWrap(True)
        video_generation_layout.addWidget(description_label)
        
        # إضافة حقل إدخال الوصف
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("الوصف:")
        prompt_layout.addWidget(prompt_label)
        
        self.video_prompt = QTextEdit()
        self.video_prompt.setPlaceholderText("أدخل وصفًا للفيديو هنا...")
        prompt_layout.addWidget(self.video_prompt)
        
        video_generation_layout.addLayout(prompt_layout)
        
        # إضافة خيارات التوليد
        options_group = QGroupBox("خيارات التوليد")
        options_layout = QVBoxLayout(options_group)
        
        # مدة الفيديو
        duration_layout = QHBoxLayout()
        duration_label = QLabel("مدة الفيديو:")
        duration_layout.addWidget(duration_label)
        
        self.duration_combo = QComboBox()
        self.duration_combo.addItem("5 ثوان", 5)
        self.duration_combo.addItem("10 ثوان", 10)
        self.duration_combo.addItem("15 ثانية", 15)
        self.duration_combo.addItem("30 ثانية", 30)
        duration_layout.addWidget(self.duration_combo)
        
        options_layout.addLayout(duration_layout)
        
        # دقة الفيديو
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("دقة الفيديو:")
        resolution_layout.addWidget(resolution_label)
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("480p", "480p")
        self.resolution_combo.addItem("720p", "720p")
        self.resolution_combo.addItem("1080p", "1080p")
        resolution_layout.addWidget(self.resolution_combo)
        
        options_layout.addLayout(resolution_layout)
        
        video_generation_layout.addWidget(options_group)
        
        # إضافة زر التوليد
        generate_button = QPushButton("توليد")
        generate_button.clicked.connect(self.generate_video)
        video_generation_layout.addWidget(generate_button)
        
        # إضافة حقل عرض الفيديو
        results_label = QLabel("الفيديو المولد:")
        video_generation_layout.addWidget(results_label)
        
        self.video_result = QLabel("سيتم عرض الفيديو هنا بعد التوليد.")
        self.video_result.setAlignment(Qt.AlignCenter)
        self.video_result.setMinimumHeight(300)
        self.video_result.setStyleSheet("border: 1px solid gray;")
        video_generation_layout.addWidget(self.video_result)
        
        # إضافة زر حفظ الفيديو
        save_button = QPushButton("حفظ الفيديو")
        save_button.clicked.connect(self.save_generated_video)
        video_generation_layout.addWidget(save_button)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(video_generation_widget, "توليد الفيديو")
        self.content_tabs.setCurrentWidget(video_generation_widget)
    
    def generate_video(self):
        """توليد الفيديو."""
        # الحصول على الوصف المدخل
        prompt = self.video_prompt.toPlainText()
        
        if not prompt:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال وصف للفيديو.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # توليد الفيديو
            self.statusBar.showMessage("جاري التوليد...")
            
            # الحصول على خيارات التوليد
            duration = self.duration_combo.currentData()
            resolution = self.resolution_combo.currentData()
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(20)
            
            # محاكاة عملية التوليد (يجب استبدالها بالتنفيذ الفعلي)
            for i in range(20, 100, 10):
                QTimer.singleShot((i - 20) * 100, lambda i=i: self.progress_bar.setValue(i))
            
            # محاكاة الفيديو المولد (يجب استبدالها بالتنفيذ الفعلي)
            self.video_result.setText(f"تم توليد فيديو بناءً على الوصف: '{prompt}'\nالمدة: {duration} ثانية\nالدقة: {resolution}")
            
            # تحديث شريط التقدم
            QTimer.singleShot(1000, lambda: self.progress_bar.setValue(100))
            self.statusBar.showMessage("تم التوليد بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error generating video: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء توليد الفيديو: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل التوليد.")
    
    def save_generated_video(self):
        """حفظ الفيديو المولد."""
        # التحقق من وجود فيديو
        if self.video_result.text().startswith("سيتم عرض الفيديو"):
            QMessageBox.warning(self, "تنبيه", "لا يوجد فيديو لحفظه.")
            return
        
        try:
            # فتح مربع حوار حفظ الملف
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "حفظ الفيديو",
                os.path.expanduser("~/generated_video.mp4"),
                "فيديو (*.mp4 *.avi *.mov)"
            )
            
            if file_path:
                # محاكاة حفظ الفيديو (يجب استبدالها بالتنفيذ الفعلي)
                self.statusBar.showMessage(f"تم حفظ الفيديو في: {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving video: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء حفظ الفيديو: {e}")
    
    def show_knowledge_visualization_tool(self):
        """عرض أداة تصور المعرفة."""
        # البحث عن علامة تبويب تصور المعرفة
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "تصور المعرفة":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت تصور المعرفة
        knowledge_visualization_widget = QWidget()
        knowledge_visualization_layout = QVBoxLayout(knowledge_visualization_widget)
        
        # إضافة عنوان
        title_label = QLabel("أداة تصور المعرفة")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        knowledge_visualization_layout.addWidget(title_label)
        
        # إضافة وصف
        description_label = QLabel("استكشف شبكة المفاهيم والعلاقات في قاعدة المعرفة.")
        description_label.setWordWrap(True)
        knowledge_visualization_layout.addWidget(description_label)
        
        # إضافة حقل البحث
        search_layout = QHBoxLayout()
        search_label = QLabel("بحث:")
        search_layout.addWidget(search_label)
        
        self.knowledge_search = QLineEdit()
        self.knowledge_search.setPlaceholderText("أدخل مفهومًا للبحث...")
        search_layout.addWidget(self.knowledge_search)
        
        search_button = QPushButton("بحث")
        search_button.clicked.connect(self.search_knowledge)
        search_layout.addWidget(search_button)
        
        knowledge_visualization_layout.addLayout(search_layout)
        
        # إضافة منطقة عرض الرسم البياني
        graph_label = QLabel("الرسم البياني للمعرفة:")
        knowledge_visualization_layout.addWidget(graph_label)
        
        self.knowledge_graph_view = QLabel("سيتم عرض الرسم البياني هنا.")
        self.knowledge_graph_view.setAlignment(Qt.AlignCenter)
        self.knowledge_graph_view.setMinimumHeight(400)
        self.knowledge_graph_view.setStyleSheet("border: 1px solid gray;")
        knowledge_visualization_layout.addWidget(self.knowledge_graph_view)
        
        # إضافة منطقة تفاصيل المفهوم
        details_label = QLabel("تفاصيل المفهوم:")
        knowledge_visualization_layout.addWidget(details_label)
        
        self.concept_details = QTextEdit()
        self.concept_details.setReadOnly(True)
        self.concept_details.setMaximumHeight(150)
        knowledge_visualization_layout.addWidget(self.concept_details)
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(knowledge_visualization_widget, "تصور المعرفة")
        self.content_tabs.setCurrentWidget(knowledge_visualization_widget)
    
    def search_knowledge(self):
        """البحث في قاعدة المعرفة."""
        # الحصول على المفهوم المدخل
        concept = self.knowledge_search.text()
        
        if not concept:
            QMessageBox.warning(self, "تنبيه", "الرجاء إدخال مفهوم للبحث.")
            return
        
        try:
            # عرض شريط التقدم
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # البحث في قاعدة المعرفة
            self.statusBar.showMessage("جاري البحث...")
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(50)
            
            # محاكاة نتائج البحث (يجب استبدالها بالتنفيذ الفعلي)
            # هنا نستخدم صورة تجريبية بدلاً من الرسم البياني الفعلي
            placeholder_image = QPixmap(600, 400)
            placeholder_image.fill(QColor(240, 240, 240))
            
            # عرض الرسم البياني
            self.knowledge_graph_view.setPixmap(placeholder_image)
            
            # عرض تفاصيل المفهوم
            concept_details = f"المفهوم: {concept}\n\n"
            concept_details += "الوصف: هذا وصف تجريبي للمفهوم المحدد.\n\n"
            concept_details += "العلاقات:\n"
            concept_details += "- علاقة 1: مفهوم آخر 1\n"
            concept_details += "- علاقة 2: مفهوم آخر 2\n"
            concept_details += "- علاقة 3: مفهوم آخر 3"
            
            self.concept_details.setPlainText(concept_details)
            
            # تحديث شريط التقدم
            self.progress_bar.setValue(100)
            self.statusBar.showMessage("تم البحث بنجاح.")
            
            # إخفاء شريط التقدم بعد فترة
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء البحث: {e}")
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("فشل البحث.")
    
    def show_settings(self):
        """عرض الإعدادات."""
        # البحث عن علامة تبويب الإعدادات
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "الإعدادات":
                # تنشيط علامة التبويب إذا كانت موجودة
                self.content_tabs.setCurrentIndex(i)
                return
        
        # إنشاء ويدجت الإعدادات
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # إضافة عنوان
        title_label = QLabel("الإعدادات")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        settings_layout.addWidget(title_label)
        
        # إضافة إعدادات المظهر
        appearance_group = QGroupBox("المظهر")
        appearance_layout = QVBoxLayout(appearance_group)
        
        # إعدادات السمة
        theme_layout = QHBoxLayout()
        theme_label = QLabel("السمة:")
        theme_layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("فاتحة", ThemeType.LIGHT)
        self.theme_combo.addItem("داكنة", ThemeType.DARK)
        self.theme_combo.addItem("النظام", ThemeType.SYSTEM)
        self.theme_combo.currentIndexChanged.connect(lambda: self.set_theme(self.theme_combo.currentData()))
        theme_layout.addWidget(self.theme_combo)
        
        appearance_layout.addLayout(theme_layout)
        
        # إعدادات نمط العرض
        view_mode_layout = QHBoxLayout()
        view_mode_label = QLabel("نمط العرض:")
        view_mode_layout.addWidget(view_mode_label)
        
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("بسيط", ViewMode.SIMPLE)
        self.view_mode_combo.addItem("متقدم", ViewMode.ADVANCED)
        self.view_mode_combo.addItem("خبير", ViewMode.EXPERT)
        self.view_mode_combo.currentIndexChanged.connect(lambda: self.set_view_mode(self.view_mode_combo.currentData()))
        view_mode_layout.addWidget(self.view_mode_combo)
        
        appearance_layout.addLayout(view_mode_layout)
        
        settings_layout.addWidget(appearance_group)
        
        # إضافة إعدادات اللغة
        language_group = QGroupBox("اللغة")
        language_layout = QVBoxLayout(language_group)
        
        # إعدادات لغة الواجهة
        ui_language_layout = QHBoxLayout()
        ui_language_label = QLabel("لغة الواجهة:")
        ui_language_layout.addWidget(ui_language_label)
        
        self.ui_language_combo = QComboBox()
        self.ui_language_combo.addItem("العربية", "ar")
        self.ui_language_combo.addItem("الإنجليزية", "en")
        ui_language_layout.addWidget(self.ui_language_combo)
        
        language_layout.addLayout(ui_language_layout)
        
        settings_layout.addWidget(language_group)
        
        # إضافة إعدادات الأداء
        performance_group = QGroupBox("الأداء")
        performance_layout = QVBoxLayout(performance_group)
        
        # إعدادات استخدام وحدة معالجة الرسومات
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("استخدام وحدة معالجة الرسومات:")
        gpu_layout.addWidget(gpu_label)
        
        self.gpu_checkbox = QCheckBox()
        gpu_layout.addWidget(self.gpu_checkbox)
        
        performance_layout.addLayout(gpu_layout)
        
        # إعدادات عدد المعالجات
        threads_layout = QHBoxLayout()
        threads_label = QLabel("عدد المعالجات:")
        threads_layout.addWidget(threads_label)
        
        self.threads_combo = QComboBox()
        for i in range(1, 9):
            self.threads_combo.addItem(str(i), i)
        threads_layout.addWidget(self.threads_combo)
        
        performance_layout.addLayout(threads_layout)
        
        settings_layout.addWidget(performance_group)
        
        # إضافة أزرار الإجراءات
        buttons_layout = QHBoxLayout()
        
        save_button = QPushButton("حفظ")
        save_button.clicked.connect(self.save_settings)
        buttons_layout.addWidget(save_button)
        
        reset_button = QPushButton("إعادة تعيين")
        reset_button.clicked.connect(self.reset_settings)
        buttons_layout.addWidget(reset_button)
        
        settings_layout.addLayout(buttons_layout)
        
        # إضافة مساحة فارغة
        settings_layout.addStretch()
        
        # إضافة علامة التبويب
        self.content_tabs.addTab(settings_widget, "الإعدادات")
        self.content_tabs.setCurrentWidget(settings_widget)
        
        # تحميل الإعدادات الحالية
        self.load_current_settings()
    
    def load_current_settings(self):
        """تحميل الإعدادات الحالية."""
        # تحميل إعدادات السمة
        theme = self.settings.value("theme", ThemeType.LIGHT.name)
        theme_index = self.theme_combo.findData(ThemeType[theme])
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        
        # تحميل إعدادات نمط العرض
        view_mode = self.settings.value("view_mode", ViewMode.SIMPLE.name)
        view_mode_index = self.view_mode_combo.findData(ViewMode[view_mode])
        if view_mode_index >= 0:
            self.view_mode_combo.setCurrentIndex(view_mode_index)
        
        # تحميل إعدادات لغة الواجهة
        ui_language = self.settings.value("ui_language", "ar")
        ui_language_index = self.ui_language_combo.findData(ui_language)
        if ui_language_index >= 0:
            self.ui_language_combo.setCurrentIndex(ui_language_index)
        
        # تحميل إعدادات استخدام وحدة معالجة الرسومات
        use_gpu = self.settings.value("use_gpu", False, type=bool)
        self.gpu_checkbox.setChecked(use_gpu)
        
        # تحميل إعدادات عدد المعالجات
        threads = self.settings.value("threads", 2, type=int)
        threads_index = self.threads_combo.findData(threads)
        if threads_index >= 0:
            self.threads_combo.setCurrentIndex(threads_index)
    
    def save_settings(self):
        """حفظ الإعدادات."""
        try:
            # حفظ إعدادات السمة
            theme = self.theme_combo.currentData().name
            self.settings.setValue("theme", theme)
            
            # حفظ إعدادات نمط العرض
            view_mode = self.view_mode_combo.currentData().name
            self.settings.setValue("view_mode", view_mode)
            
            # حفظ إعدادات لغة الواجهة
            ui_language = self.ui_language_combo.currentData()
            self.settings.setValue("ui_language", ui_language)
            
            # حفظ إعدادات استخدام وحدة معالجة الرسومات
            use_gpu = self.gpu_checkbox.isChecked()
            self.settings.setValue("use_gpu", use_gpu)
            
            # حفظ إعدادات عدد المعالجات
            threads = self.threads_combo.currentData()
            self.settings.setValue("threads", threads)
            
            # تطبيق الإعدادات
            self.apply_settings()
            
            # عرض رسالة نجاح
            self.statusBar.showMessage("تم حفظ الإعدادات بنجاح.")
        
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء حفظ الإعدادات: {e}")
    
    def reset_settings(self):
        """إعادة تعيين الإعدادات."""
        try:
            # إعادة تعيين إعدادات السمة
            self.theme_combo.setCurrentIndex(self.theme_combo.findData(ThemeType.LIGHT))
            
            # إعادة تعيين إعدادات نمط العرض
            self.view_mode_combo.setCurrentIndex(self.view_mode_combo.findData(ViewMode.SIMPLE))
            
            # إعادة تعيين إعدادات لغة الواجهة
            self.ui_language_combo.setCurrentIndex(self.ui_language_combo.findData("ar"))
            
            # إعادة تعيين إعدادات استخدام وحدة معالجة الرسومات
            self.gpu_checkbox.setChecked(False)
            
            # إعادة تعيين إعدادات عدد المعالجات
            self.threads_combo.setCurrentIndex(self.threads_combo.findData(2))
            
            # عرض رسالة نجاح
            self.statusBar.showMessage("تم إعادة تعيين الإعدادات.")
        
        except Exception as e:
            self.logger.error(f"Error resetting settings: {e}")
            QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء إعادة تعيين الإعدادات: {e}")
    
    def apply_settings(self):
        """تطبيق الإعدادات."""
        # تطبيق إعدادات السمة
        theme = ThemeType[self.settings.value("theme", ThemeType.LIGHT.name)]
        self.set_theme(theme)
        
        # تطبيق إعدادات نمط العرض
        view_mode = ViewMode[self.settings.value("view_mode", ViewMode.SIMPLE.name)]
        self.set_view_mode(view_mode)
    
    def set_theme(self, theme: ThemeType):
        """
        تعيين سمة التطبيق.
        
        Args:
            theme: نوع السمة
        """
        if theme == ThemeType.LIGHT:
            # تطبيق السمة الفاتحة
            self.setStyleSheet("")
            self.statusBar.showMessage("تم تطبيق السمة الفاتحة.")
        
        elif theme == ThemeType.DARK:
            # تطبيق السمة الداكنة
            self.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                
                QMenuBar, QMenu {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                
                QMenuBar::item:selected, QMenu::item:selected {
                    background-color: #3d3d3d;
                }
                
                QToolBar {
                    background-color: #1e1e1e;
                    border: none;
                }
                
                QPushButton {
                    background-color: #0078d7;
                    color: #ffffff;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                
                QPushButton:hover {
                    background-color: #1e88e5;
                }
                
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                
                QLineEdit, QTextEdit, QComboBox {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    border: 1px solid #5d5d5d;
                    border-radius: 3px;
                    padding: 2px;
                }
                
                QTabWidget::pane {
                    border: 1px solid #5d5d5d;
                }
                
                QTabBar::tab {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #5d5d5d;
                    padding: 5px 10px;
                    margin-right: 2px;
                }
                
                QTabBar::tab:selected {
                    background-color: #3d3d3d;
                }
                
                QTreeView, QListView {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    border: 1px solid #5d5d5d;
                }
                
                QTreeView::item:selected, QListView::item:selected {
                    background-color: #0078d7;
                }
                
                QGroupBox {
                    border: 1px solid #5d5d5d;
                    border-radius: 3px;
                    margin-top: 10px;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 5px;
                }
                
                QStatusBar {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                
                QProgressBar {
                    border: 1px solid #5d5d5d;
                    border-radius: 3px;
                    background-color: #3d3d3d;
                    text-align: center;
                    color: #ffffff;
                }
                
                QProgressBar::chunk {
                    background-color: #0078d7;
                    width: 10px;
                }
            """)
            self.statusBar.showMessage("تم تطبيق السمة الداكنة.")
        
        elif theme == ThemeType.SYSTEM:
            # تطبيق سمة النظام
            self.setStyleSheet("")
            self.statusBar.showMessage("تم تطبيق سمة النظام.")
    
    def set_view_mode(self, view_mode: ViewMode):
        """
        تعيين نمط العرض.
        
        Args:
            view_mode: نمط العرض
        """
        if view_mode == ViewMode.SIMPLE:
            # تطبيق نمط العرض البسيط
            self.statusBar.showMessage("تم تطبيق نمط العرض البسيط.")
        
        elif view_mode == ViewMode.ADVANCED:
            # تطبيق نمط العرض المتقدم
            self.statusBar.showMessage("تم تطبيق نمط العرض المتقدم.")
        
        elif view_mode == ViewMode.EXPERT:
            # تطبيق نمط العرض الخبير
            self.statusBar.showMessage("تم تطبيق نمط العرض الخبير.")
    
    def load_settings(self):
        """تحميل الإعدادات المحفوظة."""
        # تحميل إعدادات السمة
        theme = self.settings.value("theme", ThemeType.LIGHT.name)
        self.set_theme(ThemeType[theme])
        
        # تحميل إعدادات نمط العرض
        view_mode = self.settings.value("view_mode", ViewMode.SIMPLE.name)
        self.set_view_mode(ViewMode[view_mode])
    
    def new_file(self):
        """إنشاء ملف جديد."""
        # إنشاء علامة تبويب جديدة
        new_tab = QWidget()
        new_tab_layout = QVBoxLayout(new_tab)
        
        # إضافة محرر النص
        text_editor = QTextEdit()
        new_tab_layout.addWidget(text_editor)
        
        # إضافة علامة التبويب
        tab_index = self.content_tabs.addTab(new_tab, "ملف جديد")
        self.content_tabs.setCurrentIndex(tab_index)
        
        # تحديث الحالة
        self.statusBar.showMessage("تم إنشاء ملف جديد.")
    
    def open_file(self):
        """فتح ملف."""
        # فتح مربع حوار اختيار الملف
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "فتح ملف",
            os.path.expanduser("~"),
            "جميع الملفات (*)"
        )
        
        if file_path:
            try:
                # قراءة محتوى الملف
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # إنشاء علامة تبويب جديدة
                new_tab = QWidget()
                new_tab_layout = QVBoxLayout(new_tab)
                
                # إضافة محرر النص
                text_editor = QTextEdit()
                text_editor.setPlainText(content)
                new_tab_layout.addWidget(text_editor)
                
                # إضافة علامة التبويب
                file_name = os.path.basename(file_path)
                tab_index = self.content_tabs.addTab(new_tab, file_name)
                self.content_tabs.setCurrentIndex(tab_index)
                
                # تحديث الحالة
                self.statusBar.showMessage(f"تم فتح الملف: {file_path}")
            
            except Exception as e:
                self.logger.error(f"Error opening file: {e}")
                QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء فتح الملف: {e}")
    
    def save_file(self):
        """حفظ الملف."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # فتح مربع حوار حفظ الملف
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "حفظ الملف",
                    os.path.expanduser("~"),
                    "جميع الملفات (*)"
                )
                
                if file_path:
                    try:
                        # كتابة المحتوى إلى الملف
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(text_editor.toPlainText())
                        
                        # تحديث اسم علامة التبويب
                        file_name = os.path.basename(file_path)
                        self.content_tabs.setTabText(self.content_tabs.currentIndex(), file_name)
                        
                        # تحديث الحالة
                        self.statusBar.showMessage(f"تم حفظ الملف: {file_path}")
                    
                    except Exception as e:
                        self.logger.error(f"Error saving file: {e}")
                        QMessageBox.critical(self, "خطأ", f"حدث خطأ أثناء حفظ الملف: {e}")
    
    def save_file_as(self):
        """حفظ الملف باسم."""
        # استخدام نفس وظيفة حفظ الملف
        self.save_file()
    
    def close_tab(self, index):
        """
        إغلاق علامة التبويب.
        
        Args:
            index: مؤشر علامة التبويب
        """
        # إغلاق علامة التبويب
        self.content_tabs.removeTab(index)
    
    def undo(self):
        """التراجع عن التغييرات."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # التراجع عن التغييرات
                text_editor.undo()
    
    def redo(self):
        """إعادة التغييرات."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # إعادة التغييرات
                text_editor.redo()
    
    def cut(self):
        """قص النص."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # قص النص
                text_editor.cut()
    
    def copy(self):
        """نسخ النص."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # نسخ النص
                text_editor.copy()
    
    def paste(self):
        """لصق النص."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # لصق النص
                text_editor.paste()
    
    def run_current_code(self):
        """تنفيذ الكود الحالي."""
        # الحصول على علامة التبويب الحالية
        current_tab = self.content_tabs.currentWidget()
        
        if current_tab:
            # البحث عن محرر النص في علامة التبويب
            text_editor = current_tab.findChild(QTextEdit)
            
            if text_editor:
                # الحصول على الكود
                code = text_editor.toPlainText()
                
                if not code:
                    QMessageBox.warning(self, "تنبيه", "لا يوجد كود للتنفيذ.")
                    return
                
                # عرض أداة تنفيذ الأكواد
                self.show_code_execution_tool()
                
                # تعيين الكود في أداة تنفيذ الأكواد
                self.code_input.setPlainText(code)
                
                # تنفيذ الكود
                self.execute_code()
    
    def show_root_extraction_tool(self):
        """عرض أداة استخراج الجذور."""
        self.show_morphology_tool()
    
    def show_syntax_analysis_tool(self):
        """عرض أداة التحليل النحوي."""
        self.show_syntax_tool()
    
    def show_rhetoric_analysis_tool(self):
        """عرض أداة التحليل البلاغي."""
        self.show_rhetoric_tool()
    
    def show_about_dialog(self):
        """عرض مربع حوار حول."""
        about_text = (
            "نظام بصيرة - الإصدار 1.0.0\n\n"
            "نظام بصيرة هو نموذج لغوي معرفي توليدي ونظام ذكاء اصطناعي مبتكر "
            "يجمع بين المعالجة الرمزية والتعلم العميق والمعزز، "
            "مع قدرات متقدمة في معالجة اللغة العربية وتوليد المحتوى.\n\n"
            "© 2025 فريق تطوير نظام بصيرة. جميع الحقوق محفوظة."
        )
        
        QMessageBox.about(self, "حول نظام بصيرة", about_text)
    
    def show_help(self):
        """عرض المساعدة."""
        help_text = (
            "مساعدة نظام بصيرة\n\n"
            "للحصول على مساعدة حول استخدام النظام، يرجى الرجوع إلى الوثائق المتاحة "
            "أو التواصل مع فريق الدعم الفني."
        )
        
        QMessageBox.information(self, "مساعدة", help_text)
    
    def closeEvent(self, event):
        """
        معالجة حدث إغلاق النافذة.
        
        Args:
            event: حدث الإغلاق
        """
        # سؤال المستخدم عن تأكيد الإغلاق
        reply = QMessageBox.question(
            self,
            "تأكيد الإغلاق",
            "هل أنت متأكد من رغبتك في إغلاق التطبيق؟",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # حفظ الإعدادات
            self.save_settings()
            
            # تنظيف الموارد
            self.cleanup_resources()
            
            # قبول حدث الإغلاق
            event.accept()
        else:
            # رفض حدث الإغلاق
            event.ignore()
    
    def cleanup_resources(self):
        """تنظيف الموارد."""
        # تنظيف موارد منفذ الأكواد
        if hasattr(self, 'code_execution_service'):
            for executor in self.code_execution_service.executors.values():
                executor.cleanup()


def main():
    """النقطة الرئيسية لتشغيل التطبيق."""
    # إنشاء تطبيق Qt
    app = QApplication(sys.argv)
    
    # تعيين اتجاه النص من اليمين إلى اليسار
    app.setLayoutDirection(Qt.RightToLeft)
    
    # إنشاء النافذة الرئيسية
    window = BasiraDesktopApp()
    window.show()
    
    # تنفيذ حلقة الأحداث
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
