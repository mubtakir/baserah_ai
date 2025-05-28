#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Brainstorm Interface for Basira System
اختبار واجهة العصف الذهني لنظام بصيرة

Simple test version of brainstorm interface to verify functionality.

Author: Basira System Development Team
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import sys
import os
import math
import random
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

try:
    from basira_simple_demo import SimpleExpertSystem
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False


class BrainstormNode:
    """عقدة في خريطة العصف الذهني"""
    
    def __init__(self, text, x, y, color="#4CAF50"):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.connections = []
        self.canvas_id = None
        self.text_id = None
        self.selected = False


class TestBrainstormInterface:
    """نسخة اختبار لواجهة العصف الذهني"""

    def __init__(self):
        """تهيئة الواجهة"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - اختبار واجهة العصف الذهني")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # تهيئة النظام
        if BASIRA_AVAILABLE:
            self.expert_system = SimpleExpertSystem()
        else:
            self.expert_system = None

        # متغيرات العصف الذهني
        self.nodes = []
        self.selected_node = None
        self.dragging = False
        self.last_pos = (0, 0)

        # إنشاء الواجهة
        self.create_interface()

    def create_interface(self):
        """إنشاء الواجهة الرئيسية"""
        # العنوان الرئيسي
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(fill=tk.X, pady=10)

        title_label = tk.Label(
            title_frame, 
            text="🧠 نظام بصيرة - واجهة العصف الذهني 🧠",
            font=('Arial', 18, 'bold'),
            fg='#2196F3',
            bg='#f0f0f0'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="استكشف الأفكار والروابط - إبداع باسل يحيى عبدالله",
            font=('Arial', 12),
            fg='#666',
            bg='#f0f0f0'
        )
        subtitle_label.pack()

        # شريط الأدوات
        self.create_toolbar()

        # المنطقة الرئيسية
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # الشريط الجانبي
        self.create_sidebar(main_frame)

        # منطقة الرسم
        self.create_canvas_area(main_frame)

        # شريط الحالة
        self.create_status_bar()

        # إنشاء خريطة افتراضية
        self.create_default_brainstorm()

    def create_toolbar(self):
        """إنشاء شريط الأدوات"""
        toolbar = tk.Frame(self.root, bg='#e0e0e0', height=40)
        toolbar.pack(fill=tk.X, padx=5, pady=2)

        # أزرار الأدوات
        tk.Button(toolbar, text="🆕 جديد", command=self.new_brainstorm,
                 bg='#4CAF50', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(toolbar, text="➕ إضافة فكرة", command=self.add_idea,
                 bg='#2196F3', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(toolbar, text="🔗 ربط أفكار", command=self.connect_ideas,
                 bg='#FF9800', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(toolbar, text="🧠 تحليل ذكي", command=self.smart_analysis,
                 bg='#9C27B0', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(toolbar, text="🎯 توسيط", command=self.center_view,
                 bg='#607D8B', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)

    def create_sidebar(self, parent):
        """إنشاء الشريط الجانبي"""
        sidebar = tk.Frame(parent, bg='#f5f5f5', width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        sidebar.pack_propagate(False)

        # معلومات الفكرة المختارة
        info_frame = tk.LabelFrame(sidebar, text="معلومات الفكرة", bg='#f5f5f5')
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(info_frame, text="النص:", bg='#f5f5f5').pack(anchor=tk.W, padx=5)
        self.idea_text_var = tk.StringVar()
        self.idea_entry = tk.Entry(info_frame, textvariable=self.idea_text_var)
        self.idea_entry.pack(fill=tk.X, padx=5, pady=2)
        self.idea_entry.bind('<Return>', self.update_idea_text)

        # أدوات التحليل
        analysis_frame = tk.LabelFrame(sidebar, text="أدوات التحليل", bg='#f5f5f5')
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(analysis_frame, text="🔍 تحليل الفكرة", 
                 command=self.analyze_idea, bg='#4CAF50', fg='white').pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(analysis_frame, text="💡 اقتراحات", 
                 command=self.suggest_ideas, bg='#2196F3', fg='white').pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(analysis_frame, text="🌐 توسيع", 
                 command=self.expand_idea, bg='#FF9800', fg='white').pack(fill=tk.X, padx=5, pady=2)

        # قائمة الأفكار
        ideas_frame = tk.LabelFrame(sidebar, text="جميع الأفكار", bg='#f5f5f5')
        ideas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ideas_listbox = tk.Listbox(ideas_frame, height=10)
        self.ideas_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ideas_listbox.bind('<Double-1>', self.on_idea_double_click)

        # نتائج التحليل
        result_frame = tk.LabelFrame(sidebar, text="نتائج التحليل", bg='#f5f5f5')
        result_frame.pack(fill=tk.X, padx=5, pady=5)

        self.result_text = tk.Text(result_frame, height=6, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, padx=5, pady=5)

    def create_canvas_area(self, parent):
        """إنشاء منطقة الرسم"""
        canvas_frame = tk.Frame(parent, bg='white')
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # الكانفاس
        self.canvas = Canvas(canvas_frame, bg='white', 
                           highlightthickness=1, highlightbackground='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ربط الأحداث
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
        self.canvas.bind('<Double-Button-1>', self.on_canvas_double_click)

    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_frame = tk.Frame(self.root, bg='#e0e0e0')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        self.status_var.set(f"مرحباً بك في العصف الذهني - {'✅ بصيرة متاحة' if BASIRA_AVAILABLE else '⚠️ بصيرة غير متاحة'}")

        status_label = tk.Label(status_frame, textvariable=self.status_var, bg='#e0e0e0')
        status_label.pack(side=tk.LEFT, padx=10)

        # معلومات إضافية
        self.info_var = tk.StringVar()
        self.info_var.set("عدد الأفكار: 0")

        info_label = tk.Label(status_frame, textvariable=self.info_var, bg='#e0e0e0')
        info_label.pack(side=tk.RIGHT, padx=10)

    def create_default_brainstorm(self):
        """إنشاء خريطة عصف ذهني افتراضية"""
        # الفكرة المركزية
        center_x, center_y = 400, 300
        central_idea = BrainstormNode("نظام بصيرة الثوري", center_x, center_y, "#2196F3")
        self.nodes.append(central_idea)

        # أفكار فرعية
        ideas = [
            "النظام المبتكر للتفاضل والتكامل",
            "التفكيك الثوري للدوال", 
            "المعادلة العامة للأشكال",
            "الذكاء الاصطناعي العربي",
            "معالجة اللغة العربية",
            "التراث الرياضي"
        ]

        colors = ["#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#795548"]

        for i, idea in enumerate(ideas):
            angle = (2 * math.pi * i) / len(ideas)
            x = center_x + 200 * math.cos(angle)
            y = center_y + 200 * math.sin(angle)
            
            node = BrainstormNode(idea, int(x), int(y), colors[i])
            node.connections.append(central_idea)
            self.nodes.append(node)

        self.draw_brainstorm()
        self.update_ideas_list()
        self.update_status()

    def draw_brainstorm(self):
        """رسم خريطة العصف الذهني"""
        self.canvas.delete("all")

        # رسم الاتصالات أولاً
        for node in self.nodes:
            for connected_node in node.connections:
                self.canvas.create_line(
                    node.x, node.y, connected_node.x, connected_node.y,
                    fill="gray", width=2, tags="connection"
                )

        # رسم العقد
        for node in self.nodes:
            self.draw_node(node)

    def draw_node(self, node):
        """رسم عقدة واحدة"""
        radius = 40
        outline_color = "red" if node.selected else "black"
        outline_width = 3 if node.selected else 2

        # رسم الدائرة
        node.canvas_id = self.canvas.create_oval(
            node.x - radius, node.y - radius,
            node.x + radius, node.y + radius,
            fill=node.color, outline=outline_color, width=outline_width
        )

        # رسم النص
        node.text_id = self.canvas.create_text(
            node.x, node.y, text=node.text,
            font=('Arial', 10, 'bold'), fill='white',
            width=radius * 2 - 10
        )

    def on_canvas_click(self, event):
        """معالجة النقر على الكانفاس"""
        # البحث عن العقدة المنقورة
        clicked_node = self.find_node_at_position(event.x, event.y)
        
        # إلغاء تحديد جميع العقد
        for node in self.nodes:
            node.selected = False
            
        if clicked_node:
            clicked_node.selected = True
            self.selected_node = clicked_node
            self.idea_text_var.set(clicked_node.text)
            self.dragging = True
            self.last_pos = (event.x, event.y)
        else:
            self.selected_node = None
            self.idea_text_var.set("")
            
        self.draw_brainstorm()

    def on_canvas_drag(self, event):
        """معالجة سحب العقد"""
        if self.dragging and self.selected_node:
            dx = event.x - self.last_pos[0]
            dy = event.y - self.last_pos[1]
            
            self.selected_node.x += dx
            self.selected_node.y += dy
            
            self.last_pos = (event.x, event.y)
            self.draw_brainstorm()

    def on_canvas_release(self, event):
        """معالجة تحرير الماوس"""
        self.dragging = False

    def on_canvas_double_click(self, event):
        """معالجة النقر المزدوج"""
        self.add_idea_at_position(event.x, event.y)

    def find_node_at_position(self, x, y):
        """البحث عن عقدة في موقع معين"""
        for node in self.nodes:
            distance = math.sqrt((x - node.x)**2 + (y - node.y)**2)
            if distance <= 40:
                return node
        return None

    def add_idea(self):
        """إضافة فكرة جديدة"""
        text = tk.simpledialog.askstring("فكرة جديدة", "أدخل النص:")
        if text:
            self.add_idea_at_position(400, 300, text)

    def add_idea_at_position(self, x, y, text=None):
        """إضافة فكرة في موقع محدد"""
        if not text:
            text = tk.simpledialog.askstring("فكرة جديدة", "أدخل النص:")
        
        if text:
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
            color = random.choice(colors)
            
            new_node = BrainstormNode(text, x, y, color)
            self.nodes.append(new_node)
            
            self.draw_brainstorm()
            self.update_ideas_list()
            self.update_status()

    def connect_ideas(self):
        """ربط الأفكار"""
        if len(self.nodes) < 2:
            messagebox.showwarning("تحذير", "تحتاج إلى فكرتين على الأقل للربط")
            return
            
        messagebox.showinfo("ربط الأفكار", "انقر على فكرتين لربطهما")

    def smart_analysis(self):
        """التحليل الذكي للأفكار"""
        if not self.expert_system:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "❌ نظام بصيرة غير متاح للتحليل الذكي")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"🧠 التحليل الذكي للعصف الذهني\n")
        self.result_text.insert(tk.END, f"📅 {datetime.now().strftime('%H:%M:%S')}\n\n")

        try:
            # تحليل الأفكار الموجودة
            ideas_text = [node.text for node in self.nodes]
            
            # محاكاة التحليل الذكي
            analysis = f"""
✅ تم تحليل {len(self.nodes)} فكرة
🔗 عدد الاتصالات: {sum(len(node.connections) for node in self.nodes)}
🎯 الفكرة المركزية: {self.nodes[0].text if self.nodes else 'غير محددة'}

💡 اقتراحات للتطوير:
• إضافة المزيد من الأفكار الفرعية
• ربط الأفكار المتشابهة
• تطوير الأفكار الأساسية

🌟 تقييم الإبداع: عالي
🧠 مستوى التفكير: متقدم
            """
            
            self.result_text.insert(tk.END, analysis)
            
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ خطأ في التحليل: {e}")

    def analyze_idea(self):
        """تحليل فكرة محددة"""
        if not self.selected_node:
            messagebox.showwarning("تحذير", "اختر فكرة أولاً")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"🔍 تحليل الفكرة: {self.selected_node.text}\n\n")

        analysis = f"""
📊 تحليل مفصل:
• النص: {self.selected_node.text}
• اللون: {self.selected_node.color}
• الموقع: ({self.selected_node.x}, {self.selected_node.y})
• الاتصالات: {len(self.selected_node.connections)}

💡 خصائص الفكرة:
• الطول: {len(self.selected_node.text)} حرف
• عدد الكلمات: {len(self.selected_node.text.split())}
• التعقيد: {'عالي' if len(self.selected_node.text) > 20 else 'متوسط'}

🌟 تقييم الأهمية: {'مرتفع' if 'نظام' in self.selected_node.text else 'متوسط'}
        """
        
        self.result_text.insert(tk.END, analysis)

    def suggest_ideas(self):
        """اقتراح أفكار جديدة"""
        if not self.selected_node:
            messagebox.showwarning("تحذير", "اختر فكرة أولاً")
            return

        suggestions = [
            "تطبيقات عملية",
            "التحديات والصعوبات", 
            "الفرص المستقبلية",
            "الأدوات المطلوبة",
            "المراجع والمصادر",
            "التطوير والتحسين"
        ]

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"💡 اقتراحات لتطوير: {self.selected_node.text}\n\n")
        
        for i, suggestion in enumerate(suggestions, 1):
            self.result_text.insert(tk.END, f"{i}. {suggestion}\n")

    def expand_idea(self):
        """توسيع الفكرة"""
        if not self.selected_node:
            messagebox.showwarning("تحذير", "اختر فكرة أولاً")
            return

        # إضافة أفكار فرعية حول الفكرة المختارة
        sub_ideas = ["تطبيق", "تحليل", "تطوير", "اختبار"]
        
        for i, sub_idea in enumerate(sub_ideas):
            angle = (2 * math.pi * i) / len(sub_ideas)
            x = self.selected_node.x + 100 * math.cos(angle)
            y = self.selected_node.y + 100 * math.sin(angle)
            
            new_text = f"{sub_idea} {self.selected_node.text}"
            new_node = BrainstormNode(new_text, int(x), int(y), "#FFC107")
            new_node.connections.append(self.selected_node)
            self.nodes.append(new_node)
        
        self.draw_brainstorm()
        self.update_ideas_list()
        self.update_status()

    def update_idea_text(self, event=None):
        """تحديث نص الفكرة"""
        if self.selected_node:
            self.selected_node.text = self.idea_text_var.get()
            self.draw_brainstorm()
            self.update_ideas_list()

    def on_idea_double_click(self, event):
        """معالجة النقر المزدوج على قائمة الأفكار"""
        selection = self.ideas_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.nodes):
                # إلغاء تحديد جميع العقد
                for node in self.nodes:
                    node.selected = False
                
                # تحديد العقدة المختارة
                self.nodes[index].selected = True
                self.selected_node = self.nodes[index]
                self.idea_text_var.set(self.selected_node.text)
                
                self.draw_brainstorm()

    def update_ideas_list(self):
        """تحديث قائمة الأفكار"""
        self.ideas_listbox.delete(0, tk.END)
        for i, node in enumerate(self.nodes):
            self.ideas_listbox.insert(tk.END, f"{i+1}. {node.text}")

    def update_status(self):
        """تحديث شريط الحالة"""
        self.info_var.set(f"عدد الأفكار: {len(self.nodes)}")

    def new_brainstorm(self):
        """إنشاء عصف ذهني جديد"""
        self.nodes.clear()
        self.selected_node = None
        self.idea_text_var.set("")
        self.canvas.delete("all")
        self.update_ideas_list()
        self.update_status()
        self.result_text.delete(1.0, tk.END)

    def center_view(self):
        """توسيط العرض"""
        if self.nodes:
            # حساب المركز
            avg_x = sum(node.x for node in self.nodes) / len(self.nodes)
            avg_y = sum(node.y for node in self.nodes) / len(self.nodes)
            
            # تحريك جميع العقد للمركز
            canvas_center_x = self.canvas.winfo_width() // 2
            canvas_center_y = self.canvas.winfo_height() // 2
            
            dx = canvas_center_x - avg_x
            dy = canvas_center_y - avg_y
            
            for node in self.nodes:
                node.x += dx
                node.y += dy
            
            self.draw_brainstorm()

    def run(self):
        """تشغيل الواجهة"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    print("🚀 بدء اختبار واجهة العصف الذهني لنظام بصيرة...")
    
    try:
        app = TestBrainstormInterface()
        app.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل واجهة العصف الذهني: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
