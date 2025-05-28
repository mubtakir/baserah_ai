#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Generation Desktop Application for Basira System
تطبيق سطح المكتب للتوليد البصري - نظام بصيرة

Desktop application for the comprehensive visual generation system
with intuitive GUI, real-time preview, and advanced controls.

تطبيق سطح مكتب للنظام البصري الشامل مع واجهة مستخدم بديهية
ومعاينة فورية وتحكم متقدم.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import json
import threading
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revolutionary_database import ShapeEntity, RevolutionaryShapeDatabase
from comprehensive_visual_system import ComprehensiveVisualSystem, ComprehensiveVisualRequest

class VisualGenerationDesktopApp:
    """تطبيق سطح المكتب للتوليد البصري الثوري"""
    
    def __init__(self):
        """تهيئة التطبيق"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - التوليد البصري الثوري")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # تهيئة النظام البصري
        print("🌟 تهيئة النظام البصري الشامل...")
        self.visual_system = ComprehensiveVisualSystem()
        self.shape_db = RevolutionaryShapeDatabase()
        
        # متغيرات التطبيق
        self.selected_shape = None
        self.generation_in_progress = False
        
        # إنشاء الواجهة
        self.create_interface()
        
        # تحميل البيانات الأولية
        self.load_shapes()
        
        print("✅ تم تهيئة تطبيق سطح المكتب بنجاح!")
    
    def create_interface(self):
        """إنشاء واجهة المستخدم"""
        
        # إنشاء الإطار الرئيسي
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # تكوين الشبكة
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # إنشاء العنوان
        self.create_header(main_frame)
        
        # إنشاء لوحة التحكم
        self.create_control_panel(main_frame)
        
        # إنشاء منطقة المعاينة والنتائج
        self.create_preview_area(main_frame)
        
        # إنشاء شريط الحالة
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """إنشاء العنوان"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="🎨 نظام بصيرة للتوليد البصري الثوري",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="إبداع باسل يحيى عبدالله من العراق/الموصل",
            font=("Arial", 10)
        )
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        features_label = ttk.Label(
            header_frame,
            text="🖼️ توليد صور + 🎬 إنشاء فيديو + 🎨 رسم متقدم + 🔬 فيزياء + 🧠 خبير",
            font=("Arial", 9)
        )
        features_label.grid(row=2, column=0, sticky=tk.W)
    
    def create_control_panel(self, parent):
        """إنشاء لوحة التحكم"""
        control_frame = ttk.LabelFrame(parent, text="لوحة التحكم", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # اختيار الشكل
        shapes_frame = ttk.LabelFrame(control_frame, text="اختيار الشكل", padding="5")
        shapes_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.shapes_listbox = tk.Listbox(shapes_frame, height=6)
        self.shapes_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.shapes_listbox.bind('<<ListboxSelect>>', self.on_shape_select)
        
        shapes_scrollbar = ttk.Scrollbar(shapes_frame, orient="vertical")
        shapes_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.shapes_listbox.config(yscrollcommand=shapes_scrollbar.set)
        shapes_scrollbar.config(command=self.shapes_listbox.yview)
        
        # نوع المخرجات
        output_frame = ttk.LabelFrame(control_frame, text="نوع المخرجات", padding="5")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.output_vars = {}
        output_types = [
            ("image", "صورة ثابتة"),
            ("artwork", "عمل فني"),
            ("video", "فيديو متحرك"),
            ("animation", "رسم متحرك")
        ]
        
        for i, (key, label) in enumerate(output_types):
            var = tk.BooleanVar(value=(key == "image"))
            self.output_vars[key] = var
            ttk.Checkbutton(output_frame, text=label, variable=var).grid(
                row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2
            )
        
        # مستوى الجودة
        quality_frame = ttk.LabelFrame(control_frame, text="مستوى الجودة", padding="5")
        quality_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.quality_var = tk.StringVar(value="high")
        quality_options = [
            ("standard", "جودة عادية (1280x720)"),
            ("high", "جودة عالية (1920x1080)"),
            ("ultra", "جودة فائقة (2560x1440)"),
            ("masterpiece", "تحفة فنية (3840x2160)")
        ]
        
        for value, label in quality_options:
            ttk.Radiobutton(
                quality_frame, text=label, variable=self.quality_var, value=value
            ).pack(anchor=tk.W, padx=5, pady=2)
        
        # النمط الفني
        style_frame = ttk.LabelFrame(control_frame, text="النمط الفني", padding="5")
        style_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.style_var = tk.StringVar(value="digital_art")
        style_combo = ttk.Combobox(style_frame, textvariable=self.style_var, state="readonly")
        style_combo['values'] = [
            "photorealistic", "digital_art", "impressionist", "watercolor",
            "oil_painting", "anime", "abstract", "sketch"
        ]
        style_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # خيارات متقدمة
        advanced_frame = ttk.LabelFrame(control_frame, text="خيارات متقدمة", padding="5")
        advanced_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.physics_var = tk.BooleanVar(value=True)
        self.expert_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(
            advanced_frame, text="محاكاة فيزيائية", variable=self.physics_var
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        ttk.Checkbutton(
            advanced_frame, text="تحليل خبير", variable=self.expert_var
        ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        # التأثيرات البصرية
        effects_frame = ttk.LabelFrame(control_frame, text="التأثيرات البصرية", padding="5")
        effects_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.effects_vars = {}
        effects = [("glow", "توهج"), ("sharpen", "حدة"), ("enhance", "تحسين"), ("neon", "نيون")]
        
        for i, (key, label) in enumerate(effects):
            var = tk.BooleanVar()
            self.effects_vars[key] = var
            ttk.Checkbutton(effects_frame, text=label, variable=var).grid(
                row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2
            )
        
        # زر التوليد
        self.generate_btn = ttk.Button(
            control_frame, text="🚀 إنشاء المحتوى البصري", 
            command=self.generate_content, style="Accent.TButton"
        )
        self.generate_btn.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # شريط التقدم
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(control_frame, text="")
        self.progress_label.grid(row=8, column=0, sticky=tk.W)
    
    def create_preview_area(self, parent):
        """إنشاء منطقة المعاينة والنتائج"""
        preview_frame = ttk.LabelFrame(parent, text="المعاينة والنتائج", padding="10")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # منطقة المعاينة
        self.preview_text = tk.Text(
            preview_frame, height=15, wrap=tk.WORD, state=tk.DISABLED
        )
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical")
        preview_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.preview_text.config(yscrollcommand=preview_scrollbar.set)
        preview_scrollbar.config(command=self.preview_text.yview)
        
        # منطقة النتائج
        results_frame = ttk.LabelFrame(preview_frame, text="النتائج المولدة", padding="5")
        results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.results_tree = ttk.Treeview(
            results_frame, columns=("type", "quality", "artistic", "path"), height=6
        )
        self.results_tree.heading("#0", text="الرقم")
        self.results_tree.heading("type", text="النوع")
        self.results_tree.heading("quality", text="الجودة")
        self.results_tree.heading("artistic", text="فني")
        self.results_tree.heading("path", text="المسار")
        
        self.results_tree.column("#0", width=50)
        self.results_tree.column("type", width=100)
        self.results_tree.column("quality", width=80)
        self.results_tree.column("artistic", width=80)
        self.results_tree.column("path", width=200)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical")
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.results_tree.yview)
        
        # أزرار النتائج
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        ttk.Button(
            buttons_frame, text="فتح الملف", command=self.open_selected_file
        ).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(
            buttons_frame, text="فتح المجلد", command=self.open_results_folder
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            buttons_frame, text="حفظ التقرير", command=self.save_report
        ).grid(row=0, column=2, padx=5)
    
    def create_status_bar(self, parent):
        """إنشاء شريط الحالة"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="جاهز للاستخدام")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # إحصائيات
        self.stats_label = ttk.Label(status_frame, text="")
        self.stats_label.grid(row=0, column=1, sticky=tk.E)
        
        self.update_stats()
    
    def load_shapes(self):
        """تحميل الأشكال المتاحة"""
        try:
            shapes = self.shape_db.get_all_shapes()
            self.shapes_listbox.delete(0, tk.END)
            
            for shape in shapes:
                display_text = f"{shape.name} ({shape.category})"
                self.shapes_listbox.insert(tk.END, display_text)
            
            self.update_preview("تم تحميل {} شكل متاح للتوليد".format(len(shapes)))
            
        except Exception as e:
            messagebox.showerror("خطأ", f"خطأ في تحميل الأشكال: {e}")
    
    def on_shape_select(self, event):
        """عند اختيار شكل"""
        selection = self.shapes_listbox.curselection()
        if selection:
            index = selection[0]
            shapes = self.shape_db.get_all_shapes()
            if index < len(shapes):
                self.selected_shape = shapes[index]
                self.update_preview(f"تم اختيار: {self.selected_shape.name}")
    
    def generate_content(self):
        """توليد المحتوى البصري"""
        if not self.selected_shape:
            messagebox.showwarning("تحذير", "يرجى اختيار شكل أولاً")
            return
        
        if self.generation_in_progress:
            messagebox.showinfo("معلومات", "التوليد قيد التنفيذ بالفعل")
            return
        
        # جمع الخيارات المحددة
        output_types = [key for key, var in self.output_vars.items() if var.get()]
        if not output_types:
            messagebox.showwarning("تحذير", "يرجى اختيار نوع مخرجات واحد على الأقل")
            return
        
        custom_effects = [key for key, var in self.effects_vars.items() if var.get()]
        
        # إنشاء طلب التوليد
        request = ComprehensiveVisualRequest(
            shape=self.selected_shape,
            output_types=output_types,
            quality_level=self.quality_var.get(),
            artistic_styles=[self.style_var.get()],
            physics_simulation=self.physics_var.get(),
            expert_analysis=self.expert_var.get(),
            custom_effects=custom_effects,
            output_resolution=self.get_resolution_from_quality(),
            animation_duration=5.0
        )
        
        # بدء التوليد في خيط منفصل
        self.start_generation(request)
    
    def start_generation(self, request):
        """بدء عملية التوليد"""
        self.generation_in_progress = True
        self.generate_btn.config(state="disabled")
        self.progress_var.set(0)
        self.progress_label.config(text="بدء التوليد...")
        
        # تشغيل التوليد في خيط منفصل
        thread = threading.Thread(target=self.run_generation, args=(request,))
        thread.daemon = True
        thread.start()
    
    def run_generation(self, request):
        """تشغيل التوليد"""
        try:
            # محاكاة التقدم
            for i in range(0, 90, 10):
                self.root.after(0, lambda p=i: self.progress_var.set(p))
                self.root.after(0, lambda: self.progress_label.config(text=f"جاري المعالجة... {self.progress_var.get():.0f}%"))
                threading.Event().wait(0.5)
            
            # تنفيذ التوليد الفعلي
            result = self.visual_system.create_comprehensive_visual_content(request)
            
            # تحديث التقدم إلى 100%
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.progress_label.config(text="اكتمل التوليد!"))
            
            # عرض النتائج
            self.root.after(0, lambda: self.display_results(result))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("خطأ", f"خطأ في التوليد: {e}"))
        finally:
            self.root.after(0, self.finish_generation)
    
    def display_results(self, result):
        """عرض النتائج"""
        if result.success:
            # تحديث منطقة المعاينة
            preview_text = f"✅ تم إنشاء المحتوى بنجاح!\n\n"
            preview_text += f"⏱️ وقت المعالجة: {result.total_processing_time:.2f} ثانية\n\n"
            
            if result.expert_analysis:
                preview_text += "🧠 تحليل الخبير:\n"
                expert = result.expert_analysis
                if expert.get("overall_score"):
                    preview_text += f"   📊 النتيجة الإجمالية: {expert['overall_score']:.2%}\n"
                if expert.get("physics_analysis"):
                    physics = expert["physics_analysis"]
                    preview_text += f"   🔬 الدقة الفيزيائية: {physics.get('physical_accuracy', 0):.2%}\n"
                preview_text += "\n"
            
            if result.recommendations:
                preview_text += "💡 التوصيات:\n"
                for rec in result.recommendations[:3]:
                    preview_text += f"   • {rec}\n"
            
            self.update_preview(preview_text)
            
            # تحديث جدول النتائج
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            for i, (content_type, path) in enumerate(result.generated_content.items(), 1):
                quality = result.quality_metrics.get(content_type, 0) * 100
                artistic = result.artistic_scores.get(content_type, 0) * 100
                
                self.results_tree.insert("", "end", text=str(i), values=(
                    self.get_type_label(content_type),
                    f"{quality:.1f}%",
                    f"{artistic:.1f}%",
                    path
                ))
            
            messagebox.showinfo("نجح", "تم إنشاء المحتوى البصري بنجاح!")
        else:
            error_msg = "فشل في إنشاء المحتوى:\n"
            if result.error_messages:
                error_msg += "\n".join(result.error_messages)
            
            self.update_preview(error_msg)
            messagebox.showerror("خطأ", "فشل في إنشاء المحتوى البصري")
    
    def finish_generation(self):
        """إنهاء عملية التوليد"""
        self.generation_in_progress = False
        self.generate_btn.config(state="normal")
        self.progress_label.config(text="")
        self.update_stats()
    
    def update_preview(self, text):
        """تحديث منطقة المعاينة"""
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(1.0, text)
        self.preview_text.config(state=tk.DISABLED)
    
    def update_stats(self):
        """تحديث الإحصائيات"""
        try:
            stats = self.visual_system.get_system_statistics()
            stats_text = f"طلبات: {stats['total_requests']} | نجح: {stats['successful_generations']} | معدل: {stats.get('success_rate', 0):.1f}%"
            self.stats_label.config(text=stats_text)
        except:
            self.stats_label.config(text="إحصائيات غير متاحة")
    
    def get_resolution_from_quality(self):
        """الحصول على الدقة من مستوى الجودة"""
        resolutions = {
            'standard': (1280, 720),
            'high': (1920, 1080),
            'ultra': (2560, 1440),
            'masterpiece': (3840, 2160)
        }
        return resolutions.get(self.quality_var.get(), (1920, 1080))
    
    def get_type_label(self, content_type):
        """الحصول على تسمية نوع المحتوى"""
        labels = {
            'image': 'صورة',
            'video': 'فيديو',
            'artwork': 'عمل فني',
            'animation': 'رسم متحرك'
        }
        return labels.get(content_type, content_type)
    
    def open_selected_file(self):
        """فتح الملف المحدد"""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            file_path = item['values'][3]
            try:
                os.startfile(file_path)  # Windows
            except:
                try:
                    os.system(f"xdg-open '{file_path}'")  # Linux
                except:
                    messagebox.showinfo("معلومات", f"مسار الملف: {file_path}")
    
    def open_results_folder(self):
        """فتح مجلد النتائج"""
        try:
            current_dir = os.getcwd()
            os.startfile(current_dir)  # Windows
        except:
            try:
                os.system(f"xdg-open '{os.getcwd()}'")  # Linux
            except:
                messagebox.showinfo("معلومات", f"مجلد النتائج: {os.getcwd()}")
    
    def save_report(self):
        """حفظ تقرير النتائج"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("تقرير نتائج التوليد البصري - نظام بصيرة\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"تاريخ التوليد: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"الشكل المحدد: {self.selected_shape.name if self.selected_shape else 'غير محدد'}\n\n")
                    
                    f.write("النتائج المولدة:\n")
                    for item in self.results_tree.get_children():
                        values = self.results_tree.item(item)['values']
                        f.write(f"- {values[0]}: {values[3]} (جودة: {values[1]}, فني: {values[2]})\n")
                
                messagebox.showinfo("نجح", f"تم حفظ التقرير في: {file_path}")
            except Exception as e:
                messagebox.showerror("خطأ", f"خطأ في حفظ التقرير: {e}")
    
    def run(self):
        """تشغيل التطبيق"""
        self.root.mainloop()


def main():
    """تشغيل تطبيق سطح المكتب"""
    print("🌟" + "="*80 + "🌟")
    print("🖥️ تطبيق سطح المكتب للتوليد البصري الثوري")
    print("🎨 نظام بصيرة - إبداع باسل يحيى عبدالله")
    print("🌟" + "="*80 + "🌟")
    
    try:
        app = VisualGenerationDesktopApp()
        app.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")


if __name__ == "__main__":
    main()
