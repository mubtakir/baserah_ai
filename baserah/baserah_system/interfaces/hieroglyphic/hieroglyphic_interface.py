#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hieroglyphic Interface for Basira System

This module implements a unique hieroglyphic-style interface that represents
system functions and data using symbolic visual elements inspired by ancient
Egyptian hieroglyphs but adapted for modern AI system interaction.

Author: Basira System Development Team
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import math

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from main import BasiraSystem
    from dream_interpretation.basira_dream_integration import create_basira_dream_system
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False


class HieroglyphicSymbol:
    """رمز هيروغليفي"""
    
    def __init__(self, name: str, meaning: str, draw_function, color: str = "black"):
        self.name = name
        self.meaning = meaning
        self.draw_function = draw_function
        self.color = color
        self.active = False
        
    def draw(self, canvas: Canvas, x: int, y: int, size: int = 50):
        """رسم الرمز على الكانفاس"""
        color = "gold" if self.active else self.color
        return self.draw_function(canvas, x, y, size, color)


class HieroglyphicInterface:
    """واجهة هيروغليفية لنظام بصيرة"""
    
    def __init__(self):
        """تهيئة الواجهة الهيروغليفية"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - الواجهة الهيروغليفية")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c1810')  # لون بني داكن يشبه البردي القديم
        
        # تهيئة النظام
        self.basira_system = None
        self.dream_system = None
        
        # إنشاء الرموز الهيروغليفية
        self.create_hieroglyphic_symbols()
        
        # إنشاء الواجهة
        self.create_interface()
        
        # تهيئة مكونات النظام
        self.initialize_system()
        
    def create_hieroglyphic_symbols(self):
        """إنشاء الرموز الهيروغليفية"""
        self.symbols = {
            'dream': HieroglyphicSymbol(
                "حلم", "تفسير الأحلام", 
                self.draw_dream_symbol, "#4a90e2"
            ),
            'code': HieroglyphicSymbol(
                "كود", "تنفيذ الأكواد", 
                self.draw_code_symbol, "#50c878"
            ),
            'image': HieroglyphicSymbol(
                "صورة", "توليد الصور", 
                self.draw_image_symbol, "#ff6b6b"
            ),
            'text': HieroglyphicSymbol(
                "نص", "معالجة النصوص", 
                self.draw_text_symbol, "#ffa500"
            ),
            'math': HieroglyphicSymbol(
                "رياضة", "حل المعادلات", 
                self.draw_math_symbol, "#9b59b6"
            ),
            'mind': HieroglyphicSymbol(
                "عقل", "العصف الذهني", 
                self.draw_mind_symbol, "#1abc9c"
            ),
            'system': HieroglyphicSymbol(
                "نظام", "مراقبة النظام", 
                self.draw_system_symbol, "#e74c3c"
            ),
            'wisdom': HieroglyphicSymbol(
                "حكمة", "المعرفة والحكمة", 
                self.draw_wisdom_symbol, "#f39c12"
            )
        }
        
    def create_interface(self):
        """إنشاء الواجهة الرئيسية"""
        # العنوان الرئيسي
        title_frame = tk.Frame(self.root, bg='#2c1810')
        title_frame.pack(fill=tk.X, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="𓂀 نظام بصيرة - الواجهة الهيروغليفية 𓂀",
            font=('Arial', 20, 'bold'),
            fg='gold',
            bg='#2c1810'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="حيث تلتقي الحكمة القديمة بالتقنية الحديثة",
            font=('Arial', 12),
            fg='#d4af37',
            bg='#2c1810'
        )
        subtitle_label.pack()
        
        # الكانفاس الرئيسي للرموز
        self.main_canvas = Canvas(
            self.root, 
            width=900, 
            height=500,
            bg='#3e2723',
            highlightthickness=2,
            highlightbackground='gold'
        )
        self.main_canvas.pack(pady=20)
        
        # ربط الأحداث
        self.main_canvas.bind("<Button-1>", self.on_symbol_click)
        self.main_canvas.bind("<Motion>", self.on_mouse_move)
        
        # منطقة المعلومات
        info_frame = tk.Frame(self.root, bg='#2c1810')
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.info_label = tk.Label(
            info_frame,
            text="اختر رمزاً لبدء التفاعل مع النظام",
            font=('Arial', 14),
            fg='#d4af37',
            bg='#2c1810',
            wraplength=800
        )
        self.info_label.pack()
        
        # شريط الحالة
        status_frame = tk.Frame(self.root, bg='#1a1a1a')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame,
            text="جاري تهيئة النظام...",
            font=('Arial', 10),
            fg='white',
            bg='#1a1a1a'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # رسم الرموز
        self.draw_all_symbols()
        
    def draw_all_symbols(self):
        """رسم جميع الرموز الهيروغليفية"""
        canvas_width = 900
        canvas_height = 500
        
        # ترتيب الرموز في شكل دائري
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        radius = 150
        
        symbols_list = list(self.symbols.values())
        num_symbols = len(symbols_list)
        
        self.symbol_positions = {}
        
        for i, symbol in enumerate(symbols_list):
            angle = (2 * math.pi * i) / num_symbols
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # رسم الرمز
            symbol_id = symbol.draw(self.main_canvas, int(x), int(y), 60)
            
            # حفظ الموقع
            self.symbol_positions[symbol.name] = {
                'x': int(x), 'y': int(y), 'symbol': symbol, 'id': symbol_id
            }
            
            # إضافة النص
            self.main_canvas.create_text(
                int(x), int(y + 80), 
                text=symbol.meaning,
                font=('Arial', 10, 'bold'),
                fill='gold'
            )
        
        # رسم الرمز المركزي (شعار بصيرة)
        self.draw_basira_logo(center_x, center_y)
        
    def draw_basira_logo(self, x: int, y: int):
        """رسم شعار بصيرة في المركز"""
        # عين بصيرة
        self.main_canvas.create_oval(
            x-30, y-20, x+30, y+20,
            fill='#d4af37', outline='gold', width=3
        )
        
        # بؤبؤ العين
        self.main_canvas.create_oval(
            x-15, y-10, x+15, y+10,
            fill='#1a1a1a', outline='white', width=2
        )
        
        # نقطة الضوء
        self.main_canvas.create_oval(
            x-5, y-5, x+5, y+5,
            fill='white'
        )
        
        # النص
        self.main_canvas.create_text(
            x, y+40,
            text="بصيرة",
            font=('Arial', 16, 'bold'),
            fill='gold'
        )
        
    # دوال رسم الرموز الهيروغليفية
    def draw_dream_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز الحلم (هلال وسحابة)"""
        # الهلال
        canvas.create_arc(
            x-size//2, y-size//2, x+size//2, y+size//2,
            start=45, extent=270, outline=color, width=3, style='arc'
        )
        
        # السحابة
        cloud_points = [
            x-size//3, y+size//4,
            x-size//4, y+size//6,
            x-size//6, y+size//6,
            x, y+size//8,
            x+size//6, y+size//6,
            x+size//4, y+size//6,
            x+size//3, y+size//4
        ]
        canvas.create_polygon(cloud_points, fill=color, outline=color)
        
        return f"dream_{x}_{y}"
        
    def draw_code_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز الكود (أقواس وخطوط)"""
        # القوس الأيسر
        canvas.create_arc(
            x-size//2, y-size//3, x-size//4, y+size//3,
            start=270, extent=180, outline=color, width=4, style='arc'
        )
        
        # القوس الأيمن
        canvas.create_arc(
            x+size//4, y-size//3, x+size//2, y+size//3,
            start=90, extent=180, outline=color, width=4, style='arc'
        )
        
        # خطوط الكود
        for i in range(3):
            y_offset = (i - 1) * size // 6
            canvas.create_line(
                x-size//6, y+y_offset, x+size//6, y+y_offset,
                fill=color, width=2
            )
            
        return f"code_{x}_{y}"
        
    def draw_image_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز الصورة (إطار مع جبل وشمس)"""
        # الإطار
        canvas.create_rectangle(
            x-size//2, y-size//2, x+size//2, y+size//2,
            outline=color, width=3
        )
        
        # الشمس
        canvas.create_oval(
            x-size//4, y-size//3, x, y-size//6,
            fill=color, outline=color
        )
        
        # الجبل
        mountain_points = [
            x-size//2, y+size//2,
            x-size//4, y,
            x, y+size//4,
            x+size//4, y-size//6,
            x+size//2, y+size//2
        ]
        canvas.create_polygon(mountain_points, fill=color, outline=color)
        
        return f"image_{x}_{y}"
        
    def draw_text_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز النص (خطوط متوازية)"""
        for i in range(4):
            y_offset = (i - 1.5) * size // 6
            line_width = size // 2 - abs(i - 1.5) * size // 8
            canvas.create_line(
                x-line_width, y+y_offset, x+line_width, y+y_offset,
                fill=color, width=3
            )
            
        return f"text_{x}_{y}"
        
    def draw_math_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز الرياضيات (معادلة ورموز)"""
        # علامة يساوي
        canvas.create_line(
            x-size//4, y-size//8, x+size//4, y-size//8,
            fill=color, width=3
        )
        canvas.create_line(
            x-size//4, y+size//8, x+size//4, y+size//8,
            fill=color, width=3
        )
        
        # علامة زائد
        canvas.create_line(
            x-size//2, y-size//3, x-size//3, y-size//3,
            fill=color, width=3
        )
        canvas.create_line(
            x-5*size//12, y-size//2, x-5*size//12, y-size//6,
            fill=color, width=3
        )
        
        # رقم
        canvas.create_text(
            x+size//3, y-size//4,
            text="∑", font=('Arial', size//3, 'bold'),
            fill=color
        )
        
        return f"math_{x}_{y}"
        
    def draw_mind_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز العقل (دماغ مع شبكة)"""
        # الدماغ
        brain_points = [
            x-size//3, y-size//4,
            x-size//2, y-size//6,
            x-size//2, y+size//6,
            x-size//4, y+size//3,
            x+size//4, y+size//3,
            x+size//2, y+size//6,
            x+size//2, y-size//6,
            x+size//3, y-size//4,
            x, y-size//2
        ]
        canvas.create_polygon(brain_points, outline=color, width=2, fill='')
        
        # الشبكة العصبية
        for i in range(3):
            for j in range(3):
                node_x = x + (i-1) * size//6
                node_y = y + (j-1) * size//6
                canvas.create_oval(
                    node_x-3, node_y-3, node_x+3, node_y+3,
                    fill=color
                )
                
        return f"mind_{x}_{y}"
        
    def draw_system_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز النظام (ترس ومؤشرات)"""
        # الترس
        num_teeth = 8
        for i in range(num_teeth):
            angle = (2 * math.pi * i) / num_teeth
            inner_x = x + (size//3) * math.cos(angle)
            inner_y = y + (size//3) * math.sin(angle)
            outer_x = x + (size//2) * math.cos(angle)
            outer_y = y + (size//2) * math.sin(angle)
            
            canvas.create_line(inner_x, inner_y, outer_x, outer_y, fill=color, width=2)
            
        # الدائرة الداخلية
        canvas.create_oval(
            x-size//4, y-size//4, x+size//4, y+size//4,
            outline=color, width=2
        )
        
        return f"system_{x}_{y}"
        
    def draw_wisdom_symbol(self, canvas: Canvas, x: int, y: int, size: int, color: str):
        """رسم رمز الحكمة (بومة أو عين حورس)"""
        # عين حورس مبسطة
        # الجفن العلوي
        canvas.create_arc(
            x-size//2, y-size//3, x+size//2, y+size//3,
            start=0, extent=180, outline=color, width=3, style='arc'
        )
        
        # العين
        canvas.create_oval(
            x-size//4, y-size//6, x+size//4, y+size//6,
            fill=color, outline=color
        )
        
        # الخط السفلي
        canvas.create_line(
            x-size//3, y+size//4, x+size//6, y+size//3,
            fill=color, width=3
        )
        
        # الخط الجانبي
        canvas.create_line(
            x+size//3, y, x+size//2, y+size//4,
            fill=color, width=3
        )
        
        return f"wisdom_{x}_{y}"
        
    def on_symbol_click(self, event):
        """معالجة النقر على الرموز"""
        x, y = event.x, event.y
        
        # البحث عن الرمز المنقور
        clicked_symbol = None
        for symbol_name, pos_data in self.symbol_positions.items():
            symbol_x, symbol_y = pos_data['x'], pos_data['y']
            distance = math.sqrt((x - symbol_x)**2 + (y - symbol_y)**2)
            
            if distance <= 40:  # نصف قطر الرمز
                clicked_symbol = pos_data['symbol']
                break
                
        if clicked_symbol:
            self.activate_symbol(clicked_symbol)
            
    def on_mouse_move(self, event):
        """معالجة حركة الماوس"""
        x, y = event.x, event.y
        
        # البحث عن الرمز تحت الماوس
        hovered_symbol = None
        for symbol_name, pos_data in self.symbol_positions.items():
            symbol_x, symbol_y = pos_data['x'], pos_data['y']
            distance = math.sqrt((x - symbol_x)**2 + (y - symbol_y)**2)
            
            if distance <= 40:
                hovered_symbol = pos_data['symbol']
                break
                
        if hovered_symbol:
            self.info_label.config(text=f"🔮 {hovered_symbol.meaning} - انقر للتفاعل")
        else:
            self.info_label.config(text="اختر رمزاً لبدء التفاعل مع النظام")
            
    def activate_symbol(self, symbol: HieroglyphicSymbol):
        """تفعيل رمز معين"""
        # إلغاء تفعيل جميع الرموز
        for s in self.symbols.values():
            s.active = False
            
        # تفعيل الرمز المختار
        symbol.active = True
        
        # إعادة رسم الرموز
        self.main_canvas.delete("all")
        self.draw_all_symbols()
        
        # تنفيذ الوظيفة المرتبطة
        self.execute_symbol_function(symbol.name)
        
    def execute_symbol_function(self, symbol_name: str):
        """تنفيذ وظيفة الرمز"""
        functions = {
            'حلم': self.open_dream_interface,
            'كود': self.open_code_interface,
            'صورة': self.open_image_interface,
            'نص': self.open_text_interface,
            'رياضة': self.open_math_interface,
            'عقل': self.open_mind_interface,
            'نظام': self.open_system_interface,
            'حكمة': self.open_wisdom_interface
        }
        
        if symbol_name in functions:
            functions[symbol_name]()
        else:
            messagebox.showinfo("معلومات", f"تم اختيار: {symbol_name}")
            
    def open_dream_interface(self):
        """فتح واجهة تفسير الأحلام"""
        messagebox.showinfo("تفسير الأحلام", "🌙 سيتم فتح واجهة تفسير الأحلام...")
        
    def open_code_interface(self):
        """فتح واجهة تنفيذ الكود"""
        messagebox.showinfo("تنفيذ الكود", "💻 سيتم فتح واجهة تنفيذ الكود...")
        
    def open_image_interface(self):
        """فتح واجهة توليد الصور"""
        messagebox.showinfo("توليد الصور", "🎨 سيتم فتح واجهة توليد الصور...")
        
    def open_text_interface(self):
        """فتح واجهة معالجة النصوص"""
        messagebox.showinfo("معالجة النصوص", "📝 سيتم فتح واجهة معالجة النصوص...")
        
    def open_math_interface(self):
        """فتح واجهة حل المعادلات"""
        messagebox.showinfo("حل المعادلات", "🧮 سيتم فتح واجهة حل المعادلات...")
        
    def open_mind_interface(self):
        """فتح واجهة العصف الذهني"""
        messagebox.showinfo("العصف الذهني", "🧠 سيتم فتح واجهة العصف الذهني...")
        
    def open_system_interface(self):
        """فتح واجهة مراقبة النظام"""
        messagebox.showinfo("مراقبة النظام", "📊 سيتم فتح واجهة مراقبة النظام...")
        
    def open_wisdom_interface(self):
        """فتح واجهة المعرفة والحكمة"""
        messagebox.showinfo("المعرفة والحكمة", "🔮 سيتم فتح واجهة المعرفة والحكمة...")
        
    def initialize_system(self):
        """تهيئة مكونات النظام"""
        try:
            if BASIRA_AVAILABLE:
                self.basira_system = BasiraSystem()
                self.dream_system = create_basira_dream_system()
                self.status_label.config(text="✅ النظام جاهز - الواجهة الهيروغليفية نشطة")
            else:
                self.status_label.config(text="⚠️ النظام جزئي - بعض المكونات غير متاحة")
        except Exception as e:
            self.status_label.config(text=f"❌ خطأ في التهيئة: {str(e)}")
            
    def run(self):
        """تشغيل الواجهة"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    interface = HieroglyphicInterface()
    interface.run()


if __name__ == "__main__":
    main()
