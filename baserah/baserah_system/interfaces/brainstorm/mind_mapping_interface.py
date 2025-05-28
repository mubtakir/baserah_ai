#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Mind Mapping and Brainstorming Interface for Basira System

This module implements an advanced mind mapping interface that visualizes
connections between ideas, concepts, and information in an interactive way.

Author: Basira System Development Team
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, colorchooser
import sys
import os
import json
import math
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from main import BasiraSystem
    from arabic_nlp.advanced_processor import ArabicNLPProcessor
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False


class MindNode:
    """عقدة في الخريطة الذهنية"""
    
    def __init__(self, text: str, x: int, y: int, level: int = 0, color: str = "#4CAF50"):
        self.text = text
        self.x = x
        self.y = y
        self.level = level
        self.color = color
        self.children = []
        self.parent = None
        self.canvas_id = None
        self.text_id = None
        self.selected = False
        self.expanded = True
        self.connections = []  # اتصالات مع عقد أخرى
        
    def add_child(self, child_node):
        """إضافة عقدة فرعية"""
        child_node.parent = self
        child_node.level = self.level + 1
        self.children.append(child_node)
        
    def remove_child(self, child_node):
        """إزالة عقدة فرعية"""
        if child_node in self.children:
            self.children.remove(child_node)
            child_node.parent = None
            
    def get_all_descendants(self):
        """الحصول على جميع العقد التابعة"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
        
    def add_connection(self, other_node, connection_type: str = "related"):
        """إضافة اتصال مع عقدة أخرى"""
        connection = {
            'node': other_node,
            'type': connection_type,
            'strength': 1.0
        }
        self.connections.append(connection)


class MindMappingInterface:
    """واجهة العصف الذهني والخرائط الذهنية المتقدمة"""
    
    def __init__(self):
        """تهيئة الواجهة"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - واجهة العصف الذهني المتقدمة")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # متغيرات النظام
        self.basira_system = None
        self.arabic_processor = None
        
        # متغيرات الخريطة الذهنية
        self.root_node = None
        self.all_nodes = []
        self.selected_node = None
        self.dragging_node = None
        self.last_click_pos = (0, 0)
        
        # إعدادات الرسم
        self.node_radius = 30
        self.font_size = 10
        self.colors = [
            "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", 
            "#F44336", "#00BCD4", "#FFEB3B", "#795548"
        ]
        self.current_color_index = 0
        
        # إنشاء الواجهة
        self.create_interface()
        
        # تهيئة النظام
        self.initialize_system()
        
    def create_interface(self):
        """إنشاء واجهة المستخدم"""
        # شريط الأدوات العلوي
        self.create_toolbar()
        
        # المنطقة الرئيسية
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # الشريط الجانبي
        self.create_sidebar(main_frame)
        
        # منطقة الرسم
        self.create_canvas_area(main_frame)
        
        # شريط الحالة
        self.create_status_bar()
        
    def create_toolbar(self):
        """إنشاء شريط الأدوات"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        
        # أزرار الأدوات الأساسية
        ttk.Button(toolbar, text="🆕 خريطة جديدة", command=self.new_mind_map).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📁 فتح", command=self.load_mind_map).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 حفظ", command=self.save_mind_map).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # أدوات التحرير
        ttk.Button(toolbar, text="➕ إضافة عقدة", command=self.add_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔗 ربط عقد", command=self.connect_nodes).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🎨 تغيير اللون", command=self.change_node_color).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="❌ حذف", command=self.delete_node).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # أدوات التحليل
        ttk.Button(toolbar, text="🧠 تحليل ذكي", command=self.smart_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍 البحث", command=self.search_nodes).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 إحصائيات", command=self.show_statistics).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # أدوات العرض
        ttk.Button(toolbar, text="🔍+ تكبير", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍- تصغير", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🎯 توسيط", command=self.center_view).pack(side=tk.LEFT, padx=2)
        
    def create_sidebar(self, parent):
        """إنشاء الشريط الجانبي"""
        sidebar_frame = ttk.Frame(parent, width=250)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        sidebar_frame.pack_propagate(False)
        
        # معلومات العقدة المختارة
        info_frame = ttk.LabelFrame(sidebar_frame, text="معلومات العقدة")
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="النص:").pack(anchor=tk.W, padx=5)
        self.node_text_var = tk.StringVar()
        self.node_text_entry = ttk.Entry(info_frame, textvariable=self.node_text_var)
        self.node_text_entry.pack(fill=tk.X, padx=5, pady=2)
        self.node_text_entry.bind('<Return>', self.update_node_text)
        
        ttk.Label(info_frame, text="المستوى:").pack(anchor=tk.W, padx=5)
        self.node_level_label = ttk.Label(info_frame, text="غير محدد")
        self.node_level_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(info_frame, text="عدد الفروع:").pack(anchor=tk.W, padx=5)
        self.node_children_label = ttk.Label(info_frame, text="0")
        self.node_children_label.pack(anchor=tk.W, padx=5)
        
        # أدوات التحليل النصي
        analysis_frame = ttk.LabelFrame(sidebar_frame, text="التحليل النصي")
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="🔍 تحليل النص", 
                  command=self.analyze_node_text).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="💡 اقتراحات", 
                  command=self.suggest_related_ideas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="🌐 توسيع الفكرة", 
                  command=self.expand_idea).pack(fill=tk.X, padx=5, pady=2)
        
        # قائمة العقد
        nodes_frame = ttk.LabelFrame(sidebar_frame, text="جميع العقد")
        nodes_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # إنشاء Treeview لعرض العقد
        self.nodes_tree = ttk.Treeview(nodes_frame, height=10)
        self.nodes_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.nodes_tree.bind('<Double-1>', self.on_tree_double_click)
        
        # شريط التمرير للقائمة
        tree_scrollbar = ttk.Scrollbar(nodes_frame, orient=tk.VERTICAL, command=self.nodes_tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.nodes_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # إعدادات العرض
        display_frame = ttk.LabelFrame(sidebar_frame, text="إعدادات العرض")
        display_frame.pack(fill=tk.X, pady=5)
        
        # خيارات العرض
        self.show_connections_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="إظهار الاتصالات", 
                       variable=self.show_connections_var,
                       command=self.refresh_canvas).pack(anchor=tk.W, padx=5)
        
        self.show_levels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="إظهار المستويات", 
                       variable=self.show_levels_var,
                       command=self.refresh_canvas).pack(anchor=tk.W, padx=5)
        
        self.animate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="تحريك العقد", 
                       variable=self.animate_var).pack(anchor=tk.W, padx=5)
        
    def create_canvas_area(self, parent):
        """إنشاء منطقة الرسم"""
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # إنشاء الكانفاس مع أشرطة التمرير
        self.canvas = tk.Canvas(
            canvas_frame, 
            bg='white', 
            scrollregion=(0, 0, 2000, 2000),
            highlightthickness=1,
            highlightbackground='gray'
        )
        
        # أشرطة التمرير
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # ترتيب العناصر
        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # ربط الأحداث
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
        self.canvas.bind('<Double-Button-1>', self.on_canvas_double_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        
        # إنشاء خريطة ذهنية افتراضية
        self.create_default_mind_map()
        
    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("مرحباً بك في واجهة العصف الذهني المتقدمة")
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)
        
        # معلومات إضافية
        self.info_var = tk.StringVar()
        self.info_var.set("عدد العقد: 0")
        
        info_label = ttk.Label(status_frame, textvariable=self.info_var)
        info_label.pack(side=tk.RIGHT, padx=10)
        
    def create_default_mind_map(self):
        """إنشاء خريطة ذهنية افتراضية"""
        # إنشاء العقدة الجذر
        center_x, center_y = 400, 300
        self.root_node = MindNode("نظام بصيرة", center_x, center_y, 0, "#2196F3")
        self.all_nodes = [self.root_node]
        
        # إضافة عقد فرعية
        main_topics = [
            "تفسير الأحلام", "معالجة اللغة", "توليد المحتوى", 
            "حل المعادلات", "الذكاء الاصطناعي", "التراث العربي"
        ]
        
        for i, topic in enumerate(main_topics):
            angle = (2 * math.pi * i) / len(main_topics)
            x = center_x + 150 * math.cos(angle)
            y = center_y + 150 * math.sin(angle)
            
            child_node = MindNode(topic, int(x), int(y), 1, self.colors[i % len(self.colors)])
            self.root_node.add_child(child_node)
            self.all_nodes.append(child_node)
            
        # رسم الخريطة
        self.draw_mind_map()
        self.update_nodes_tree()
        self.update_status()
        
    def draw_mind_map(self):
        """رسم الخريطة الذهنية"""
        self.canvas.delete("all")
        
        # رسم الاتصالات أولاً
        if self.show_connections_var.get():
            self.draw_connections()
            
        # رسم العقد
        for node in self.all_nodes:
            self.draw_node(node)
            
        # رسم الخطوط بين العقد الأساسية
        self.draw_hierarchy_lines()
        
    def draw_node(self, node: MindNode):
        """رسم عقدة واحدة"""
        # تحديد اللون حسب الحالة
        fill_color = node.color
        outline_color = "black"
        
        if node.selected:
            outline_color = "red"
            outline_width = 3
        else:
            outline_width = 2
            
        # رسم الدائرة
        radius = self.node_radius + (5 if node.level == 0 else 0)
        node.canvas_id = self.canvas.create_oval(
            node.x - radius, node.y - radius,
            node.x + radius, node.y + radius,
            fill=fill_color, outline=outline_color, width=outline_width
        )
        
        # رسم النص
        font_size = self.font_size + (2 if node.level == 0 else 0)
        node.text_id = self.canvas.create_text(
            node.x, node.y,
            text=node.text,
            font=('Arial', font_size, 'bold' if node.level == 0 else 'normal'),
            fill='white' if self.is_dark_color(node.color) else 'black',
            width=radius * 2 - 10
        )
        
        # رسم مؤشر المستوى
        if self.show_levels_var.get() and node.level > 0:
            level_text = str(node.level)
            self.canvas.create_text(
                node.x + radius - 8, node.y - radius + 8,
                text=level_text,
                font=('Arial', 8, 'bold'),
                fill='white',
                tags="level_indicator"
            )
            
    def draw_connections(self):
        """رسم الاتصالات بين العقد"""
        for node in self.all_nodes:
            for connection in node.connections:
                other_node = connection['node']
                connection_type = connection['type']
                
                # تحديد لون ونمط الخط حسب نوع الاتصال
                if connection_type == "related":
                    line_color = "blue"
                    line_style = (5, 5)  # خط متقطع
                elif connection_type == "opposite":
                    line_color = "red"
                    line_style = (10, 5)
                else:
                    line_color = "green"
                    line_style = ()
                    
                # رسم الخط
                self.canvas.create_line(
                    node.x, node.y, other_node.x, other_node.y,
                    fill=line_color, width=2, dash=line_style,
                    tags="connection"
                )
                
    def draw_hierarchy_lines(self):
        """رسم خطوط التسلسل الهرمي"""
        for node in self.all_nodes:
            for child in node.children:
                self.canvas.create_line(
                    node.x, node.y, child.x, child.y,
                    fill="gray", width=2, tags="hierarchy"
                )
                
    def is_dark_color(self, color: str) -> bool:
        """تحديد ما إذا كان اللون داكناً"""
        # تحويل اللون إلى RGB وحساب السطوع
        if color.startswith('#'):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return brightness < 128
