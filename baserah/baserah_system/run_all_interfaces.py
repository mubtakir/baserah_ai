#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run All Interfaces - Basira System
تشغيل جميع الواجهات - نظام بصيرة

This script provides a unified launcher for all Basira System interfaces.
يوفر هذا الملف مشغل موحد لجميع واجهات نظام بصيرة.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 3.0.0 - "Revolutionary Integration"
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
import webbrowser
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

try:
    from basira_simple_demo import SimpleExpertSystem
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False


class BasiraInterfaceLauncher:
    """مشغل موحد لجميع واجهات نظام بصيرة"""

    def __init__(self):
        """تهيئة المشغل"""
        self.root = tk.Tk()
        self.root.title("نظام بصيرة - مشغل الواجهات الموحد")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # متغيرات النظام
        self.running_processes = {}
        self.web_server_port = 5000

        # تهيئة النظام
        if BASIRA_AVAILABLE:
            self.expert_system = SimpleExpertSystem()
        else:
            self.expert_system = None

        # إنشاء الواجهة
        self.create_interface()

    def create_interface(self):
        """إنشاء واجهة المشغل"""
        # العنوان الرئيسي
        self.create_header()
        
        # معلومات النظام
        self.create_system_info()
        
        # أزرار الواجهات
        self.create_interface_buttons()
        
        # منطقة الحالة والسجلات
        self.create_status_area()
        
        # شريط الحالة
        self.create_status_bar()

    def create_header(self):
        """إنشاء العنوان الرئيسي"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🌟 نظام بصيرة - مشغل الواجهات الموحد 🌟",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="إبداع باسل يحيى عبدالله - العراق/الموصل | Created by Basil Yahya Abdullah - Iraq/Mosul",
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack()

        version_label = tk.Label(
            header_frame,
            text='الإصدار 3.0.0 - "التكامل الثوري" | Version 3.0.0 - "Revolutionary Integration"',
            font=('Arial', 10),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        version_label.pack()

    def create_system_info(self):
        """إنشاء معلومات النظام"""
        info_frame = tk.LabelFrame(self.root, text="معلومات النظام", bg='#f0f0f0', font=('Arial', 12, 'bold'))
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        info_text = f"""
🧠 حالة نظام بصيرة: {'✅ متاح ويعمل' if BASIRA_AVAILABLE else '❌ غير متاح'}
📅 تاريخ اليوم: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🐍 إصدار Python: {sys.version.split()[0]}
💻 نظام التشغيل: {os.name}
🔧 المسار الحالي: {os.getcwd()}

🌟 الأنظمة الرياضية المتاحة:
• النظام المبتكر للتفاضل والتكامل (تكامل = V × A، تفاضل = D × A)
• النظام الثوري لتفكيك الدوال (A = x.dA - ∫x.d2A)
• المعادلة العامة للأشكال
• نظام الخبير/المستكشف المتكامل
        """

        info_label = tk.Label(info_frame, text=info_text, justify=tk.LEFT, bg='#f0f0f0', font=('Arial', 10))
        info_label.pack(padx=10, pady=5)

    def create_interface_buttons(self):
        """إنشاء أزرار الواجهات"""
        buttons_frame = tk.LabelFrame(self.root, text="الواجهات المتاحة", bg='#f0f0f0', font=('Arial', 12, 'bold'))
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)

        # الصف الأول من الأزرار
        row1_frame = tk.Frame(buttons_frame, bg='#f0f0f0')
        row1_frame.pack(fill=tk.X, padx=5, pady=5)

        # واجهة سطح المكتب
        desktop_btn = tk.Button(
            row1_frame,
            text="🖥️ واجهة سطح المكتب\nDesktop Interface",
            command=self.launch_desktop_interface,
            bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
            height=3, width=20
        )
        desktop_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # واجهة الويب
        web_btn = tk.Button(
            row1_frame,
            text="🌐 واجهة الويب\nWeb Interface",
            command=self.launch_web_interface,
            bg='#2ecc71', fg='white', font=('Arial', 11, 'bold'),
            height=3, width=20
        )
        web_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # الصف الثاني من الأزرار
        row2_frame = tk.Frame(buttons_frame, bg='#f0f0f0')
        row2_frame.pack(fill=tk.X, padx=5, pady=5)

        # الواجهة الهيروغلوفية
        hieroglyphic_btn = tk.Button(
            row2_frame,
            text="📜 الواجهة الهيروغلوفية\nHieroglyphic Interface",
            command=self.launch_hieroglyphic_interface,
            bg='#f39c12', fg='white', font=('Arial', 11, 'bold'),
            height=3, width=20
        )
        hieroglyphic_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # واجهة العصف الذهني
        brainstorm_btn = tk.Button(
            row2_frame,
            text="🧠 واجهة العصف الذهني\nBrainstorm Interface",
            command=self.launch_brainstorm_interface,
            bg='#9b59b6', fg='white', font=('Arial', 11, 'bold'),
            height=3, width=20
        )
        brainstorm_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # أزرار إضافية
        extra_frame = tk.Frame(buttons_frame, bg='#f0f0f0')
        extra_frame.pack(fill=tk.X, padx=5, pady=5)

        # تشغيل جميع الواجهات
        all_btn = tk.Button(
            extra_frame,
            text="🚀 تشغيل جميع الواجهات\nLaunch All Interfaces",
            command=self.launch_all_interfaces,
            bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
            height=2
        )
        all_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # إيقاف جميع الواجهات
        stop_btn = tk.Button(
            extra_frame,
            text="🛑 إيقاف جميع الواجهات\nStop All Interfaces",
            command=self.stop_all_interfaces,
            bg='#95a5a6', fg='white', font=('Arial', 11, 'bold'),
            height=2
        )
        stop_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def create_status_area(self):
        """إنشاء منطقة الحالة والسجلات"""
        status_frame = tk.LabelFrame(self.root, text="حالة الواجهات والسجلات", bg='#f0f0f0', font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # منطقة النص للسجلات
        self.log_text = tk.Text(status_frame, height=10, wrap=tk.WORD, bg='#2c3e50', fg='#ecf0f1', font=('Courier', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # شريط التمرير
        scrollbar = tk.Scrollbar(status_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # رسالة ترحيب
        welcome_msg = f"""
🌟 مرحباً بك في نظام بصيرة - مشغل الواجهات الموحد 🌟
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚀 اختر الواجهة التي تريد تشغيلها:
🖥️ واجهة سطح المكتب - للاستخدام المحلي
🌐 واجهة الويب - للوصول عبر المتصفح
📜 الواجهة الهيروغلوفية - تجربة تفاعلية فريدة
🧠 واجهة العصف الذهني - لاستكشاف الأفكار

💡 يمكنك تشغيل عدة واجهات في نفس الوقت!
        """
        self.log_text.insert(tk.END, welcome_msg)

    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_bar_frame = tk.Frame(self.root, bg='#34495e')
        status_bar_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        self.status_var.set("جاهز للتشغيل - Ready to Launch")

        status_label = tk.Label(status_bar_frame, textvariable=self.status_var, bg='#34495e', fg='white', font=('Arial', 10))
        status_label.pack(side=tk.LEFT, padx=10)

        # معلومات الواجهات النشطة
        self.active_interfaces_var = tk.StringVar()
        self.active_interfaces_var.set("الواجهات النشطة: 0")

        active_label = tk.Label(status_bar_frame, textvariable=self.active_interfaces_var, bg='#34495e', fg='white', font=('Arial', 10))
        active_label.pack(side=tk.RIGHT, padx=10)

    def log_message(self, message):
        """إضافة رسالة إلى السجل"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update()

    def launch_desktop_interface(self):
        """تشغيل واجهة سطح المكتب"""
        self.log_message("🖥️ بدء تشغيل واجهة سطح المكتب...")
        self.status_var.set("تشغيل واجهة سطح المكتب...")
        
        try:
            process = subprocess.Popen([sys.executable, "baserah_system/test_desktop_interface.py"])
            self.running_processes['desktop'] = process
            self.log_message("✅ تم تشغيل واجهة سطح المكتب بنجاح!")
            self.update_active_interfaces()
        except Exception as e:
            self.log_message(f"❌ خطأ في تشغيل واجهة سطح المكتب: {e}")
            messagebox.showerror("خطأ", f"فشل في تشغيل واجهة سطح المكتب:\n{e}")

    def launch_web_interface(self):
        """تشغيل واجهة الويب"""
        self.log_message("🌐 بدء تشغيل واجهة الويب...")
        self.status_var.set("تشغيل واجهة الويب...")
        
        try:
            # تشغيل خادم الويب في خيط منفصل
            def run_web_server():
                process = subprocess.Popen([sys.executable, "baserah_system/test_web_interface.py"])
                self.running_processes['web'] = process
                
            web_thread = threading.Thread(target=run_web_server, daemon=True)
            web_thread.start()
            
            self.log_message("✅ تم تشغيل خادم الويب!")
            self.log_message(f"🌐 الواجهة متاحة على: http://localhost:{self.web_server_port}")
            
            # فتح المتصفح بعد ثانيتين
            self.root.after(2000, lambda: webbrowser.open(f"http://localhost:{self.web_server_port}"))
            self.log_message("🌐 سيتم فتح المتصفح تلقائياً...")
            
            self.update_active_interfaces()
        except Exception as e:
            self.log_message(f"❌ خطأ في تشغيل واجهة الويب: {e}")
            messagebox.showerror("خطأ", f"فشل في تشغيل واجهة الويب:\n{e}")

    def launch_hieroglyphic_interface(self):
        """تشغيل الواجهة الهيروغلوفية"""
        self.log_message("📜 بدء تشغيل الواجهة الهيروغلوفية...")
        self.status_var.set("تشغيل الواجهة الهيروغلوفية...")
        
        try:
            process = subprocess.Popen([sys.executable, "baserah_system/test_hieroglyphic_interface.py"])
            self.running_processes['hieroglyphic'] = process
            self.log_message("✅ تم تشغيل الواجهة الهيروغلوفية بنجاح!")
            self.update_active_interfaces()
        except Exception as e:
            self.log_message(f"❌ خطأ في تشغيل الواجهة الهيروغلوفية: {e}")
            messagebox.showerror("خطأ", f"فشل في تشغيل الواجهة الهيروغلوفية:\n{e}")

    def launch_brainstorm_interface(self):
        """تشغيل واجهة العصف الذهني"""
        self.log_message("🧠 بدء تشغيل واجهة العصف الذهني...")
        self.status_var.set("تشغيل واجهة العصف الذهني...")
        
        try:
            process = subprocess.Popen([sys.executable, "baserah_system/test_brainstorm_interface.py"])
            self.running_processes['brainstorm'] = process
            self.log_message("✅ تم تشغيل واجهة العصف الذهني بنجاح!")
            self.update_active_interfaces()
        except Exception as e:
            self.log_message(f"❌ خطأ في تشغيل واجهة العصف الذهني: {e}")
            messagebox.showerror("خطأ", f"فشل في تشغيل واجهة العصف الذهني:\n{e}")

    def launch_all_interfaces(self):
        """تشغيل جميع الواجهات"""
        self.log_message("🚀 بدء تشغيل جميع الواجهات...")
        self.status_var.set("تشغيل جميع الواجهات...")
        
        # تشغيل كل واجهة مع تأخير قصير
        self.launch_desktop_interface()
        self.root.after(1000, self.launch_hieroglyphic_interface)
        self.root.after(2000, self.launch_brainstorm_interface)
        self.root.after(3000, self.launch_web_interface)
        
        self.log_message("🎉 تم تشغيل جميع الواجهات!")

    def stop_all_interfaces(self):
        """إيقاف جميع الواجهات"""
        self.log_message("🛑 بدء إيقاف جميع الواجهات...")
        self.status_var.set("إيقاف جميع الواجهات...")
        
        for interface_name, process in self.running_processes.items():
            try:
                process.terminate()
                self.log_message(f"🛑 تم إيقاف {interface_name}")
            except Exception as e:
                self.log_message(f"❌ خطأ في إيقاف {interface_name}: {e}")
        
        self.running_processes.clear()
        self.update_active_interfaces()
        self.log_message("✅ تم إيقاف جميع الواجهات")
        self.status_var.set("جاهز للتشغيل")

    def update_active_interfaces(self):
        """تحديث عدد الواجهات النشطة"""
        # تنظيف العمليات المنتهية
        active_processes = {}
        for name, process in self.running_processes.items():
            if process.poll() is None:  # العملية ما زالت تعمل
                active_processes[name] = process
        
        self.running_processes = active_processes
        count = len(self.running_processes)
        self.active_interfaces_var.set(f"الواجهات النشطة: {count}")

    def on_closing(self):
        """معالجة إغلاق التطبيق"""
        if self.running_processes:
            result = messagebox.askyesno(
                "إغلاق التطبيق",
                "هناك واجهات نشطة. هل تريد إيقافها وإغلاق التطبيق؟"
            )
            if result:
                self.stop_all_interfaces()
                self.root.destroy()
        else:
            self.root.destroy()

    def run(self):
        """تشغيل المشغل"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    print("🌟" + "="*80 + "🌟")
    print("🚀 نظام بصيرة - مشغل الواجهات الموحد")
    print("🚀 Basira System - Unified Interface Launcher")
    print("🌟 إبداع باسل يحيى عبدالله - العراق/الموصل")
    print("🌟 Created by Basil Yahya Abdullah - Iraq/Mosul")
    print("🌟" + "="*80 + "🌟")
    
    try:
        launcher = BasiraInterfaceLauncher()
        launcher.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل المشغل: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
