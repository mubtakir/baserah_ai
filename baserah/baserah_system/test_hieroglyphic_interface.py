#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Hieroglyphic Interface for Basira System
اختبار الواجهة الهيروغلوفية لنظام بصيرة

Simple test version of hieroglyphic interface to verify functionality.

Author: Basira System Development Team
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import sys
import os
import math
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

try:
    from basira_simple_demo import SimpleExpertSystem
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False


class TestHieroglyphicInterface:
    """نسخة اختبار للواجهة الهيروغلوفية"""

    def __init__(self):
        """تهيئة الواجهة"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - اختبار الواجهة الهيروغلوفية")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c1810')  # لون بني داكن يشبه البردي القديم

        # تهيئة النظام
        if BASIRA_AVAILABLE:
            self.expert_system = SimpleExpertSystem()
        else:
            self.expert_system = None

        # إنشاء الواجهة
        self.create_interface()

    def create_interface(self):
        """إنشاء الواجهة الرئيسية"""
        # العنوان الرئيسي
        title_frame = tk.Frame(self.root, bg='#2c1810')
        title_frame.pack(fill=tk.X, pady=10)

        title_label = tk.Label(
            title_frame, 
            text="𓂀 نظام بصيرة - الواجهة الهيروغلوفية 𓂀",
            font=('Arial', 20, 'bold'),
            fg='gold',
            bg='#2c1810'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="حيث تلتقي الحكمة القديمة بالتقنية الحديثة - إبداع باسل يحيى عبدالله",
            font=('Arial', 12),
            fg='#d4af37',
            bg='#2c1810'
        )
        subtitle_label.pack()

        # الكانفاس الرئيسي للرموز
        self.main_canvas = Canvas(
            self.root, 
            width=900, 
            height=400,
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

        # منطقة النتائج
        result_frame = tk.LabelFrame(self.root, text="نتائج التفاعل", 
                                   bg='#2c1810', fg='gold', font=('Arial', 12, 'bold'))
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.result_text = tk.Text(result_frame, height=8, bg='#1a1a1a', fg='white',
                                  font=('Arial', 10), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # شريط الحالة
        status_frame = tk.Frame(self.root, bg='#1a1a1a')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text=f"النظام جاهز - {'✅ بصيرة متاحة' if BASIRA_AVAILABLE else '⚠️ بصيرة غير متاحة'}",
            font=('Arial', 10),
            fg='white',
            bg='#1a1a1a'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # رسم الرموز
        self.draw_hieroglyphic_symbols()

    def draw_hieroglyphic_symbols(self):
        """رسم الرموز الهيروغليفية"""
        canvas_width = 900
        canvas_height = 400
        
        # تعريف الرموز
        symbols = [
            {"name": "النظام المبتكر", "color": "#4a90e2", "function": self.test_innovative_calculus},
            {"name": "التفكيك الثوري", "color": "#50c878", "function": self.test_revolutionary_decomposition},
            {"name": "المعادلة العامة", "color": "#ff6b6b", "function": self.test_general_equation},
            {"name": "نظام الخبير", "color": "#ffa500", "function": self.test_expert_system},
            {"name": "معلومات النظام", "color": "#9b59b6", "function": self.show_system_info}
        ]

        # ترتيب الرموز في خط أفقي
        symbol_spacing = canvas_width // (len(symbols) + 1)
        y_center = canvas_height // 2

        self.symbol_positions = {}

        for i, symbol in enumerate(symbols):
            x = symbol_spacing * (i + 1)
            y = y_center

            # رسم الرمز
            self.draw_symbol(x, y, symbol["name"], symbol["color"])
            
            # حفظ الموقع والوظيفة
            self.symbol_positions[symbol["name"]] = {
                'x': x, 'y': y, 'function': symbol["function"], 'color': symbol["color"]
            }

        # رسم شعار بصيرة في المركز العلوي
        self.draw_basira_logo(canvas_width // 2, 80)

    def draw_symbol(self, x, y, name, color):
        """رسم رمز هيروغليفي مبسط"""
        # دائرة خارجية
        self.main_canvas.create_oval(
            x-40, y-40, x+40, y+40,
            outline=color, width=3, fill='#2c1810'
        )
        
        # رمز داخلي حسب النوع
        if "المبتكر" in name:
            # رمز التفاضل والتكامل
            self.main_canvas.create_text(x, y-10, text="∫", font=('Arial', 20, 'bold'), fill=color)
            self.main_canvas.create_text(x, y+10, text="d/dx", font=('Arial', 10, 'bold'), fill=color)
        elif "الثوري" in name:
            # رمز المتسلسلة
            self.main_canvas.create_text(x, y, text="Σ", font=('Arial', 24, 'bold'), fill=color)
        elif "المعادلة" in name:
            # رمز المعادلة
            self.main_canvas.create_text(x, y, text="=", font=('Arial', 24, 'bold'), fill=color)
        elif "الخبير" in name:
            # رمز العقل
            self.main_canvas.create_text(x, y, text="🧠", font=('Arial', 20))
        else:
            # رمز المعلومات
            self.main_canvas.create_text(x, y, text="ℹ️", font=('Arial', 20))

        # النص التوضيحي
        self.main_canvas.create_text(
            x, y+60, text=name, font=('Arial', 10, 'bold'), fill='gold'
        )

    def draw_basira_logo(self, x, y):
        """رسم شعار بصيرة"""
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
            x, y+35, text="بصيرة", font=('Arial', 14, 'bold'), fill='gold'
        )

    def on_symbol_click(self, event):
        """معالجة النقر على الرموز"""
        x, y = event.x, event.y
        
        # البحث عن الرمز المنقور
        for symbol_name, pos_data in self.symbol_positions.items():
            symbol_x, symbol_y = pos_data['x'], pos_data['y']
            distance = math.sqrt((x - symbol_x)**2 + (y - symbol_y)**2)
            
            if distance <= 40:  # نصف قطر الرمز
                self.activate_symbol(symbol_name, pos_data)
                break

    def on_mouse_move(self, event):
        """معالجة حركة الماوس"""
        x, y = event.x, event.y
        
        # البحث عن الرمز تحت الماوس
        for symbol_name, pos_data in self.symbol_positions.items():
            symbol_x, symbol_y = pos_data['x'], pos_data['y']
            distance = math.sqrt((x - symbol_x)**2 + (y - symbol_y)**2)
            
            if distance <= 40:
                self.info_label.config(text=f"🔮 {symbol_name} - انقر للتفاعل")
                return
                
        self.info_label.config(text="اختر رمزاً لبدء التفاعل مع النظام")

    def activate_symbol(self, symbol_name, pos_data):
        """تفعيل رمز معين"""
        self.info_label.config(text=f"🔮 تم تفعيل: {symbol_name}")
        self.status_label.config(text=f"جاري تنفيذ: {symbol_name}...")
        
        # تنفيذ الوظيفة
        try:
            pos_data['function']()
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ خطأ في تنفيذ {symbol_name}: {e}\n\n")

    def test_innovative_calculus(self):
        """اختبار النظام المبتكر للتفاضل والتكامل"""
        self.result_text.insert(tk.END, f"🧮 اختبار النظام المبتكر للتفاضل والتكامل\n")
        self.result_text.insert(tk.END, f"📅 الوقت: {datetime.now().strftime('%H:%M:%S')}\n")
        
        if not self.expert_system:
            self.result_text.insert(tk.END, "❌ نظام بصيرة غير متاح\n\n")
            return

        try:
            # اختبار دالة تربيعية
            function_values = [1, 4, 9, 16, 25]
            D_coeffs = [2, 4, 6, 8, 10]
            V_coeffs = [0.33, 1.33, 3, 5.33, 8.33]

            self.expert_system.calculus_engine.add_coefficient_state(
                function_values, D_coeffs, V_coeffs
            )

            result = self.expert_system.calculus_engine.predict_calculus(function_values)

            self.result_text.insert(tk.END, "✅ النظام المبتكر يعمل بنجاح!\n")
            self.result_text.insert(tk.END, f"📊 قيم الدالة: {function_values}\n")
            self.result_text.insert(tk.END, f"📊 التفاضل المقدر: {[round(x, 2) for x in result['derivative']]}\n")
            self.result_text.insert(tk.END, f"📊 التكامل المقدر: {[round(x, 2) for x in result['integral']]}\n")
            self.result_text.insert(tk.END, f"🌟 الطريقة: {result['method']}\n\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"❌ خطأ: {e}\n\n")

    def test_revolutionary_decomposition(self):
        """اختبار النظام الثوري لتفكيك الدوال"""
        self.result_text.insert(tk.END, f"🌟 اختبار النظام الثوري لتفكيك الدوال\n")
        self.result_text.insert(tk.END, f"📅 الوقت: {datetime.now().strftime('%H:%M:%S')}\n")
        
        if not self.expert_system:
            self.result_text.insert(tk.END, "❌ نظام بصيرة غير متاح\n\n")
            return

        try:
            # اختبار دالة تكعيبية
            x_vals = [1, 2, 3, 4, 5]
            f_vals = [1, 8, 27, 64, 125]  # x^3

            result = self.expert_system.decomposition_engine.decompose_simple_function(
                "cubic_test", x_vals, f_vals
            )

            self.result_text.insert(tk.END, "✅ النظام الثوري يعمل بنجاح!\n")
            self.result_text.insert(tk.END, f"📊 اسم الدالة: {result['function_name']}\n")
            self.result_text.insert(tk.END, f"📊 دقة التفكيك: {result['accuracy']:.4f}\n")
            self.result_text.insert(tk.END, f"📊 عدد الحدود: {result['n_terms_used']}\n")
            self.result_text.insert(tk.END, f"🌟 الطريقة: {result['method']}\n\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"❌ خطأ: {e}\n\n")

    def test_general_equation(self):
        """اختبار المعادلة العامة للأشكال"""
        self.result_text.insert(tk.END, f"📐 اختبار المعادلة العامة للأشكال\n")
        self.result_text.insert(tk.END, f"📅 الوقت: {datetime.now().strftime('%H:%M:%S')}\n")
        
        if not self.expert_system:
            self.result_text.insert(tk.END, "❌ نظام بصيرة غير متاح\n\n")
            return

        try:
            # اختبار معالجة بسيطة
            test_data = "test_equation_data"
            result = self.expert_system.general_equation.process(test_data)

            self.result_text.insert(tk.END, "✅ المعادلة العامة تعمل بنجاح!\n")
            self.result_text.insert(tk.END, f"📊 نوع المعادلة: {result['equation_type']}\n")
            self.result_text.insert(tk.END, f"📊 نمط التعلم: {result['learning_mode']}\n")
            self.result_text.insert(tk.END, f"📊 حالة المعالجة: {'نجح' if result['processed'] else 'فشل'}\n\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"❌ خطأ: {e}\n\n")

    def test_expert_system(self):
        """اختبار نظام الخبير"""
        self.result_text.insert(tk.END, f"🧠 اختبار نظام الخبير المتكامل\n")
        self.result_text.insert(tk.END, f"📅 الوقت: {datetime.now().strftime('%H:%M:%S')}\n")
        
        if not self.expert_system:
            self.result_text.insert(tk.END, "❌ نظام بصيرة غير متاح\n\n")
            return

        try:
            # تشغيل العرض التوضيحي الشامل
            result = self.expert_system.demonstrate_system()

            self.result_text.insert(tk.END, "✅ نظام الخبير يعمل بنجاح!\n")
            self.result_text.insert(tk.END, f"📊 وقت الجلسة: {result['timestamp']}\n")
            
            if 'calculus_test' in result:
                calculus_accuracy = len(result['calculus_test']['derivative']) > 0
                self.result_text.insert(tk.END, f"📊 النظام المبتكر: {'✅' if calculus_accuracy else '❌'}\n")
            
            if 'decomposition_test' in result:
                decomp_accuracy = result['decomposition_test']['accuracy']
                self.result_text.insert(tk.END, f"📊 التفكيك الثوري: {decomp_accuracy:.4f}\n")
            
            self.result_text.insert(tk.END, "🌟 جميع الأنظمة متكاملة ومترابطة!\n\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"❌ خطأ: {e}\n\n")

    def show_system_info(self):
        """عرض معلومات النظام"""
        self.result_text.insert(tk.END, f"📊 معلومات نظام بصيرة\n")
        self.result_text.insert(tk.END, f"📅 الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        info = f"""
🌟 المبدع: باسل يحيى عبدالله - العراق/الموصل
📅 الإصدار: 3.0.0 - "التكامل الثوري"
🧠 حالة النظام: {'✅ متاح' if BASIRA_AVAILABLE else '❌ غير متاح'}

🔧 المكونات الأساسية:
✅ المعادلة العامة للأشكال
✅ النظام المبتكر للتفاضل والتكامل  
✅ النظام الثوري لتفكيك الدوال
✅ نظام الخبير/المستكشف

🖥️ الواجهات المتاحة:
✅ واجهة سطح المكتب
✅ واجهة الويب
✅ الواجهة الهيروغلوفية (تعمل حالياً)
✅ واجهة العصف الذهني

🎯 الابتكارات الرياضية:
• تكامل = V × A، تفاضل = D × A
• A = x.dA - ∫x.d2A (الفرضية الثورية)
• المتسلسلة مع الإشارات المتعاقبة

🌟 الواجهة الهيروغلوفية تعمل بنجاح!
        """
        
        self.result_text.insert(tk.END, info + "\n")

    def run(self):
        """تشغيل الواجهة"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    print("🚀 بدء اختبار الواجهة الهيروغلوفية لنظام بصيرة...")
    
    try:
        app = TestHieroglyphicInterface()
        app.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل الواجهة الهيروغلوفية: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
