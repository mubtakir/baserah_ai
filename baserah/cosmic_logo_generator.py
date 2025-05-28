#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مولد شعار نظام بصيرة الكوني المتكامل
Cosmic Baserah System Logo Generator

Author: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0 - Complete Logo System
"""

import tkinter as tk
from tkinter import Canvas, PhotoImage
import math
import colorsys

class CosmicLogoGenerator:
    """مولد الشعار الكوني لنظام بصيرة"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.create_logo_canvas()
        self.cosmic_colors = self.define_cosmic_colors()
        
    def setup_window(self):
        """إعداد نافذة التصميم"""
        self.root.title("🌟 مولد شعار نظام بصيرة الكوني 🌟")
        self.root.geometry("1000x800")
        self.root.configure(bg="#0a0a0a")
        
    def define_cosmic_colors(self):
        """تعريف الألوان الكونية"""
        return {
            "cosmic_blue": "#1e3a8a",      # أزرق كوني عميق
            "cosmic_purple": "#7c3aed",    # بنفسجي كوني
            "cosmic_gold": "#f59e0b",      # ذهبي كوني
            "cosmic_silver": "#e5e7eb",    # فضي كوني
            "basil_green": "#10b981",      # أخضر باسل
            "wisdom_orange": "#f97316",    # برتقالي الحكمة
            "star_white": "#ffffff",       # أبيض النجوم
            "space_black": "#000000",      # أسود الفضاء
            "energy_cyan": "#06b6d4",      # سماوي الطاقة
            "mystic_violet": "#8b5cf6"     # بنفسجي صوفي
        }
    
    def create_logo_canvas(self):
        """إنشاء لوحة الرسم"""
        self.canvas = Canvas(
            self.root, 
            width=800, 
            height=600,
            bg="#000011",
            highlightthickness=0
        )
        self.canvas.pack(pady=20)
        
    def draw_cosmic_background(self):
        """رسم الخلفية الكونية"""
        # خلفية متدرجة كونية
        for i in range(600):
            color_intensity = int(255 * (1 - i/600) * 0.1)
            color = f"#{color_intensity:02x}{color_intensity:02x}{min(color_intensity + 20, 255):02x}"
            self.canvas.create_line(0, i, 800, i, fill=color, width=1)
        
        # نجوم متناثرة
        import random
        for _ in range(100):
            x = random.randint(0, 800)
            y = random.randint(0, 600)
            size = random.randint(1, 3)
            brightness = random.choice(["#ffffff", "#f0f0f0", "#e0e0e0", "#d0d0d0"])
            self.canvas.create_oval(x, y, x+size, y+size, fill=brightness, outline="")
    
    def draw_central_symbol(self):
        """رسم الرمز المركزي"""
        center_x, center_y = 400, 300
        
        # الدائرة الخارجية - تمثل الكون
        self.canvas.create_oval(
            center_x - 150, center_y - 150,
            center_x + 150, center_y + 150,
            outline=self.cosmic_colors["cosmic_gold"],
            width=4,
            fill=""
        )
        
        # الدائرة الوسطى - تمثل الحكمة
        self.canvas.create_oval(
            center_x - 100, center_y - 100,
            center_x + 100, center_y + 100,
            outline=self.cosmic_colors["cosmic_purple"],
            width=3,
            fill=""
        )
        
        # الدائرة الداخلية - تمثل الإبداع
        self.canvas.create_oval(
            center_x - 60, center_y - 60,
            center_x + 60, center_y + 60,
            fill=self.cosmic_colors["basil_green"],
            outline=self.cosmic_colors["cosmic_gold"],
            width=2
        )
        
        # رمز العين الكونية في المركز
        self.draw_cosmic_eye(center_x, center_y)
        
        # الأشعة الكونية
        self.draw_cosmic_rays(center_x, center_y)
        
        # الرموز العربية الكونية
        self.draw_arabic_symbols(center_x, center_y)
    
    def draw_cosmic_eye(self, x, y):
        """رسم العين الكونية"""
        # العين الخارجية
        self.canvas.create_oval(
            x - 30, y - 15,
            x + 30, y + 15,
            fill=self.cosmic_colors["cosmic_blue"],
            outline=self.cosmic_colors["star_white"],
            width=2
        )
        
        # البؤبؤ
        self.canvas.create_oval(
            x - 15, y - 10,
            x + 15, y + 10,
            fill=self.cosmic_colors["space_black"],
            outline=""
        )
        
        # نقطة الضوء
        self.canvas.create_oval(
            x - 5, y - 5,
            x + 5, y + 5,
            fill=self.cosmic_colors["star_white"],
            outline=""
        )
    
    def draw_cosmic_rays(self, x, y):
        """رسم الأشعة الكونية"""
        for i in range(8):
            angle = i * math.pi / 4
            start_x = x + 70 * math.cos(angle)
            start_y = y + 70 * math.sin(angle)
            end_x = x + 120 * math.cos(angle)
            end_y = y + 120 * math.sin(angle)
            
            # شعاع رئيسي
            self.canvas.create_line(
                start_x, start_y, end_x, end_y,
                fill=self.cosmic_colors["cosmic_gold"],
                width=3
            )
            
            # شعاع ثانوي
            mid_x = x + 95 * math.cos(angle)
            mid_y = y + 95 * math.sin(angle)
            self.canvas.create_oval(
                mid_x - 3, mid_y - 3,
                mid_x + 3, mid_y + 3,
                fill=self.cosmic_colors["wisdom_orange"],
                outline=""
            )
    
    def draw_arabic_symbols(self, x, y):
        """رسم الرموز العربية الكونية"""
        # حرف ب (باسل) في الأعلى
        self.canvas.create_text(
            x, y - 180,
            text="ب",
            font=("Traditional Arabic", 24, "bold"),
            fill=self.cosmic_colors["cosmic_gold"]
        )
        
        # حرف ص (بصيرة) في اليمين
        self.canvas.create_text(
            x + 180, y,
            text="ص",
            font=("Traditional Arabic", 20, "bold"),
            fill=self.cosmic_colors["cosmic_purple"]
        )
        
        # حرف ر (بصيرة) في الأسفل
        self.canvas.create_text(
            x, y + 180,
            text="ر",
            font=("Traditional Arabic", 20, "bold"),
            fill=self.cosmic_colors["basil_green"]
        )
        
        # حرف ة (بصيرة) في اليسار
        self.canvas.create_text(
            x - 180, y,
            text="ة",
            font=("Traditional Arabic", 20, "bold"),
            fill=self.cosmic_colors["wisdom_orange"]
        )
    
    def draw_logo_text(self):
        """رسم نص الشعار"""
        # العنوان الرئيسي بالعربية
        self.canvas.create_text(
            400, 100,
            text="نظام بصيرة الكوني",
            font=("Traditional Arabic", 28, "bold"),
            fill=self.cosmic_colors["cosmic_gold"]
        )
        
        # العنوان بالإنجليزية
        self.canvas.create_text(
            400, 130,
            text="Cosmic Baserah System",
            font=("Arial", 16, "bold"),
            fill=self.cosmic_colors["cosmic_silver"]
        )
        
        # الشعار السفلي
        self.canvas.create_text(
            400, 500,
            text="إبداع باسل يحيى عبدالله",
            font=("Traditional Arabic", 18, "bold"),
            fill=self.cosmic_colors["cosmic_purple"]
        )
        
        self.canvas.create_text(
            400, 525,
            text="من العراق/الموصل إلى الكون",
            font=("Traditional Arabic", 12),
            fill=self.cosmic_colors["basil_green"]
        )
        
        # الشعار الفلسفي
        self.canvas.create_text(
            400, 560,
            text="حيث تلتقي الحكمة بالتقنية",
            font=("Traditional Arabic", 14, "italic"),
            fill=self.cosmic_colors["wisdom_orange"]
        )
    
    def draw_decorative_elements(self):
        """رسم العناصر الزخرفية"""
        # زخارف إسلامية في الزوايا
        corners = [(100, 100), (700, 100), (100, 500), (700, 500)]
        
        for corner_x, corner_y in corners:
            # نجمة ثمانية
            self.draw_eight_pointed_star(corner_x, corner_y, 20)
            
        # خط زخرفي علوي
        self.draw_decorative_border(50, 50, 750, 50)
        
        # خط زخرفي سفلي
        self.draw_decorative_border(50, 550, 750, 550)
    
    def draw_eight_pointed_star(self, x, y, size):
        """رسم نجمة ثمانية الرؤوس"""
        points = []
        for i in range(16):
            angle = i * math.pi / 8
            if i % 2 == 0:
                radius = size
            else:
                radius = size * 0.4
            
            point_x = x + radius * math.cos(angle)
            point_y = y + radius * math.sin(angle)
            points.extend([point_x, point_y])
        
        self.canvas.create_polygon(
            points,
            fill=self.cosmic_colors["mystic_violet"],
            outline=self.cosmic_colors["cosmic_gold"],
            width=1
        )
    
    def draw_decorative_border(self, x1, y1, x2, y2):
        """رسم حدود زخرفية"""
        # خط رئيسي
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=self.cosmic_colors["cosmic_gold"],
            width=2
        )
        
        # نقاط زخرفية
        for i in range(int((x2 - x1) / 50)):
            point_x = x1 + i * 50
            self.canvas.create_oval(
                point_x - 3, y1 - 3,
                point_x + 3, y1 + 3,
                fill=self.cosmic_colors["cosmic_purple"],
                outline=""
            )
    
    def create_complete_logo(self):
        """إنشاء الشعار الكامل"""
        print("🎨 بدء إنشاء شعار نظام بصيرة الكوني...")
        
        # رسم الطبقات بالترتيب
        self.draw_cosmic_background()
        print("   ✅ تم رسم الخلفية الكونية")
        
        self.draw_decorative_elements()
        print("   ✅ تم رسم العناصر الزخرفية")
        
        self.draw_central_symbol()
        print("   ✅ تم رسم الرمز المركزي")
        
        self.draw_logo_text()
        print("   ✅ تم إضافة النصوص")
        
        print("🌟 تم إنشاء الشعار بنجاح!")
        
        # إضافة أزرار التحكم
        self.add_control_buttons()
    
    def add_control_buttons(self):
        """إضافة أزرار التحكم"""
        button_frame = tk.Frame(self.root, bg="#0a0a0a")
        button_frame.pack(pady=10)
        
        # زر حفظ الشعار
        save_btn = tk.Button(
            button_frame,
            text="💾 حفظ الشعار",
            command=self.save_logo,
            bg="#10b981",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        save_btn.pack(side=tk.LEFT, padx=10)
        
        # زر إنشاء متغيرات
        variants_btn = tk.Button(
            button_frame,
            text="🎨 متغيرات الشعار",
            command=self.create_logo_variants,
            bg="#7c3aed",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        variants_btn.pack(side=tk.LEFT, padx=10)
        
        # زر معاينة
        preview_btn = tk.Button(
            button_frame,
            text="👁️ معاينة",
            command=self.preview_logo,
            bg="#f59e0b",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        preview_btn.pack(side=tk.LEFT, padx=10)
    
    def save_logo(self):
        """حفظ الشعار"""
        try:
            # حفظ كصورة PostScript
            self.canvas.postscript(file="cosmic_baserah_logo.eps")
            print("✅ تم حفظ الشعار كملف EPS")
            
            # عرض رسالة نجاح
            success_window = tk.Toplevel(self.root)
            success_window.title("✅ تم الحفظ")
            success_window.geometry("400x200")
            success_window.configure(bg="#10b981")
            
            tk.Label(
                success_window,
                text="🎉 تم حفظ الشعار بنجاح!",
                font=("Arial", 16, "bold"),
                bg="#10b981",
                fg="white"
            ).pack(pady=20)
            
            tk.Label(
                success_window,
                text="الملف: cosmic_baserah_logo.eps",
                font=("Arial", 12),
                bg="#10b981",
                fg="white"
            ).pack(pady=10)
            
        except Exception as e:
            print(f"❌ خطأ في حفظ الشعار: {e}")
    
    def create_logo_variants(self):
        """إنشاء متغيرات الشعار"""
        variants_window = tk.Toplevel(self.root)
        variants_window.title("🎨 متغيرات الشعار")
        variants_window.geometry("600x400")
        variants_window.configure(bg="#1f2937")
        
        tk.Label(
            variants_window,
            text="🎨 متغيرات شعار نظام بصيرة الكوني",
            font=("Arial", 16, "bold"),
            bg="#1f2937",
            fg="#f59e0b"
        ).pack(pady=20)
        
        variants_info = """
🌟 المتغيرات المتاحة:

1. 🎯 الشعار الكامل - للاستخدام الرسمي
2. 🔷 الرمز المركزي فقط - للأيقونات
3. 📝 النص فقط - للعناوين
4. 🌙 النسخة الليلية - خلفية داكنة
5. ☀️ النسخة النهارية - خلفية فاتحة
6. 🎨 نسخة أحادية اللون - للطباعة
7. 🌈 نسخة ملونة - للوسائط الرقمية
8. 📱 نسخة مصغرة - للتطبيقات

🎯 الاستخدامات:
• المواقع الإلكترونية
• التطبيقات المحمولة  
• المطبوعات الرسمية
• العروض التقديمية
• وسائل التواصل الاجتماعي
• البضائع الترويجية
        """
        
        tk.Label(
            variants_window,
            text=variants_info,
            font=("Arial", 10),
            bg="#1f2937",
            fg="white",
            justify=tk.LEFT
        ).pack(padx=20, pady=10)
    
    def preview_logo(self):
        """معاينة الشعار"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("👁️ معاينة الشعار")
        preview_window.geometry("500x600")
        preview_window.configure(bg="white")
        
        # معاينة على خلفية بيضاء
        preview_canvas = Canvas(
            preview_window,
            width=450,
            height=400,
            bg="white"
        )
        preview_canvas.pack(pady=20)
        
        # رسم نسخة مبسطة للمعاينة
        self.draw_simple_logo_preview(preview_canvas)
        
        tk.Label(
            preview_window,
            text="معاينة الشعار على خلفية بيضاء",
            font=("Arial", 12),
            bg="white"
        ).pack(pady=10)
    
    def draw_simple_logo_preview(self, canvas):
        """رسم معاينة مبسطة للشعار"""
        center_x, center_y = 225, 200
        
        # الدائرة الخارجية
        canvas.create_oval(
            center_x - 80, center_y - 80,
            center_x + 80, center_y + 80,
            outline="#f59e0b",
            width=3
        )
        
        # الدائرة الداخلية
        canvas.create_oval(
            center_x - 40, center_y - 40,
            center_x + 40, center_y + 40,
            fill="#10b981",
            outline="#f59e0b",
            width=2
        )
        
        # النص
        canvas.create_text(
            center_x, center_y - 120,
            text="نظام بصيرة الكوني",
            font=("Arial", 16, "bold"),
            fill="#1e3a8a"
        )
        
        canvas.create_text(
            center_x, center_y + 120,
            text="إبداع باسل يحيى عبدالله",
            font=("Arial", 12),
            fill="#7c3aed"
        )
    
    def run(self):
        """تشغيل مولد الشعار"""
        self.create_complete_logo()
        self.root.mainloop()

def main():
    """الدالة الرئيسية"""
    print("🌟" + "="*60 + "🌟")
    print("🎨 مولد شعار نظام بصيرة الكوني المتكامل")
    print("🌟 إبداع باسل يحيى عبدالله من العراق/الموصل")
    print("🌟" + "="*60 + "🌟")
    
    logo_generator = CosmicLogoGenerator()
    logo_generator.run()

if __name__ == "__main__":
    main()
