#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Structure Visualizer for Basira System
مصور هيكلية المشروع لنظام بصيرة

Interactive visualization of the project structure and dependencies.
تصور تفاعلي لهيكلية المشروع والتبعيات.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

try:
    from arabic_text_handler import fix_arabic_text, fix_title_text, fix_label_text
    ARABIC_HANDLER_AVAILABLE = True
except ImportError:
    ARABIC_HANDLER_AVAILABLE = False
    def fix_arabic_text(text): return text
    def fix_title_text(text): return text
    def fix_label_text(text): return text


class ProjectStructureVisualizer:
    """مصور هيكلية المشروع"""

    def __init__(self):
        """تهيئة المصور"""
        self.root = tk.Tk()
        self.root.title(fix_title_text("نظام بصيرة - مصور هيكلية المشروع"))
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # بيانات هيكلية المشروع
        self.project_structure = self.build_project_structure()
        
        # إنشاء الواجهة
        self.create_interface()

    def build_project_structure(self):
        """بناء بيانات هيكلية المشروع"""
        return {
            "نظام بصيرة (Basira System)": {
                "type": "root",
                "description": "النظام الثوري للذكاء الاصطناعي والرياضيات",
                "children": {
                    "📁 baserah_system/": {
                        "type": "core_folder",
                        "description": "النواة الأساسية للنظام",
                        "children": {
                            "📄 basira_simple_demo.py": {
                                "type": "core_file",
                                "description": "النواة الرئيسية - قلب النظام",
                                "importance": "critical"
                            },
                            "📄 arabic_text_handler.py": {
                                "type": "utility_file", 
                                "description": "معالج النصوص العربية",
                                "importance": "high"
                            },
                            "📄 run_all_interfaces.py": {
                                "type": "launcher_file",
                                "description": "مشغل الواجهات الموحد",
                                "importance": "high"
                            },
                            "📁 interfaces/": {
                                "type": "interfaces_folder",
                                "description": "الواجهات التفاعلية",
                                "children": {
                                    "📁 desktop/": {
                                        "type": "interface_folder",
                                        "description": "واجهة سطح المكتب (tkinter)"
                                    },
                                    "📁 web/": {
                                        "type": "interface_folder", 
                                        "description": "واجهة الويب (Flask + HTML)"
                                    },
                                    "📁 hieroglyphic/": {
                                        "type": "interface_folder",
                                        "description": "الواجهة الهيروغلوفية"
                                    },
                                    "📁 brainstorm/": {
                                        "type": "interface_folder",
                                        "description": "واجهة العصف الذهني"
                                    }
                                }
                            },
                            "📁 core/": {
                                "type": "math_folder",
                                "description": "الأنظمة الرياضية الثورية",
                                "children": {
                                    "📄 general_shape_equation.py": {
                                        "type": "math_file",
                                        "description": "المعادلة العامة للأشكال",
                                        "innovation": "revolutionary"
                                    },
                                    "📄 innovative_calculus.py": {
                                        "type": "math_file",
                                        "description": "النظام المبتكر: تكامل = V × A",
                                        "innovation": "revolutionary"
                                    },
                                    "📄 revolutionary_decomposition.py": {
                                        "type": "math_file",
                                        "description": "التفكيك الثوري: A = x.dA - ∫x.d2A",
                                        "innovation": "revolutionary"
                                    },
                                    "📄 expert_explorer_system.py": {
                                        "type": "math_file",
                                        "description": "نظام الخبير/المستكشف",
                                        "innovation": "advanced"
                                    }
                                }
                            }
                        }
                    },
                    "📄 START_BASIRA_SYSTEM.py": {
                        "type": "entry_file",
                        "description": "نقطة البداية السريعة للنظام",
                        "importance": "high"
                    },
                    "📄 install_arabic_support.py": {
                        "type": "setup_file",
                        "description": "تثبيت دعم النصوص العربية",
                        "importance": "medium"
                    },
                    "📁 baserah/": {
                        "type": "legacy_folder",
                        "description": "الملفات التراثية والمراجع الأولية",
                        "children": {
                            "📄 tkamul.py": {
                                "type": "legacy_file",
                                "description": "النظام المبتكر الأصلي"
                            }
                        }
                    },
                    "📁 ai_mathematical/": {
                        "type": "reference_folder",
                        "description": "الأفكار الرياضية الأولية"
                    }
                }
            }
        }

    def create_interface(self):
        """إنشاء الواجهة الرئيسية"""
        # العنوان
        self.create_header()
        
        # المنطقة الرئيسية
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # الشجرة التفاعلية
        self.create_tree_view(main_frame)
        
        # منطقة التفاصيل
        self.create_details_panel(main_frame)
        
        # شريط الحالة
        self.create_status_bar()

    def create_header(self):
        """إنشاء العنوان"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text=fix_title_text("🗺️ مصور هيكلية نظام بصيرة 🗺️"),
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text=fix_label_text("دليل تفاعلي للمطورين لفهم هيكلية المشروع"),
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack()

    def create_tree_view(self, parent):
        """إنشاء عرض الشجرة التفاعلي"""
        tree_frame = ttk.LabelFrame(parent, text=fix_label_text("هيكلية المشروع"))
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # إنشاء Treeview
        self.tree = ttk.Treeview(tree_frame, height=25)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # شريط التمرير
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # ربط الأحداث
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        # بناء الشجرة
        self.build_tree()

    def create_details_panel(self, parent):
        """إنشاء لوحة التفاصيل"""
        details_frame = ttk.LabelFrame(parent, text=fix_label_text("تفاصيل العنصر المختار"))
        details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # معلومات العنصر
        info_frame = ttk.Frame(details_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(info_frame, text=fix_label_text("الاسم:"), font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w')
        self.name_label = tk.Label(info_frame, text="", font=('Arial', 10))
        self.name_label.grid(row=0, column=1, sticky='w', padx=(10, 0))

        tk.Label(info_frame, text=fix_label_text("النوع:"), font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w')
        self.type_label = tk.Label(info_frame, text="", font=('Arial', 10))
        self.type_label.grid(row=1, column=1, sticky='w', padx=(10, 0))

        tk.Label(info_frame, text=fix_label_text("الأهمية:"), font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w')
        self.importance_label = tk.Label(info_frame, text="", font=('Arial', 10))
        self.importance_label.grid(row=2, column=1, sticky='w', padx=(10, 0))

        # الوصف
        desc_frame = ttk.LabelFrame(details_frame, text=fix_label_text("الوصف"))
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.description_text = scrolledtext.ScrolledText(desc_frame, height=10, wrap=tk.WORD)
        self.description_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # العلاقات والتبعيات
        relations_frame = ttk.LabelFrame(details_frame, text=fix_label_text("العلاقات والتبعيات"))
        relations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.relations_text = scrolledtext.ScrolledText(relations_frame, height=8, wrap=tk.WORD)
        self.relations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_frame = tk.Frame(self.root, bg='#34495e')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        self.status_var.set(fix_label_text("جاهز - اختر عنصر من الشجرة لعرض تفاصيله"))

        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               bg='#34495e', fg='white', font=('Arial', 10))
        status_label.pack(side=tk.LEFT, padx=10)

        # معلومات إضافية
        info_label = tk.Label(status_frame, 
                             text=fix_label_text(f"آخر تحديث: {datetime.now().strftime('%Y-%m-%d')}"),
                             bg='#34495e', fg='white', font=('Arial', 10))
        info_label.pack(side=tk.RIGHT, padx=10)

    def build_tree(self):
        """بناء شجرة المشروع"""
        # مسح الشجرة
        for item in self.tree.get_children():
            self.tree.delete(item)

        # بناء الشجرة من البيانات
        self.add_tree_items("", self.project_structure)

    def add_tree_items(self, parent, data):
        """إضافة عناصر للشجرة"""
        for name, info in data.items():
            # تحديد الأيقونة حسب النوع
            icon = self.get_type_icon(info.get('type', 'unknown'))
            display_name = f"{icon} {name}"
            
            # إضافة العنصر
            item_id = self.tree.insert(parent, 'end', text=display_name, 
                                     values=(info.get('type', ''), info.get('description', '')))
            
            # حفظ البيانات مع العنصر
            self.tree.set(item_id, 'data', str(info))
            
            # إضافة العناصر الفرعية
            if 'children' in info:
                self.add_tree_items(item_id, info['children'])

    def get_type_icon(self, item_type):
        """الحصول على أيقونة حسب نوع العنصر"""
        icons = {
            'root': '🌟',
            'core_folder': '🏗️',
            'core_file': '⚙️',
            'interfaces_folder': '🖥️',
            'interface_folder': '📱',
            'math_folder': '🧮',
            'math_file': '📐',
            'utility_file': '🔧',
            'launcher_file': '🚀',
            'entry_file': '🎯',
            'setup_file': '⚙️',
            'legacy_folder': '📚',
            'legacy_file': '📜',
            'reference_folder': '📖'
        }
        return icons.get(item_type, '📄')

    def on_tree_select(self, event):
        """معالجة اختيار عنصر من الشجرة"""
        selection = self.tree.selection()
        if not selection:
            return

        item_id = selection[0]
        item_text = self.tree.item(item_id, 'text')
        
        # استخراج البيانات
        try:
            data_str = self.tree.set(item_id, 'data')
            data = eval(data_str) if data_str != 'None' else {}
        except:
            data = {}

        # تحديث التفاصيل
        self.update_details(item_text, data)

    def update_details(self, name, data):
        """تحديث لوحة التفاصيل"""
        # تنظيف الاسم من الأيقونة
        clean_name = name.split(' ', 1)[1] if ' ' in name else name
        
        # تحديث المعلومات الأساسية
        self.name_label.config(text=fix_arabic_text(clean_name))
        self.type_label.config(text=self.get_type_description(data.get('type', 'unknown')))
        self.importance_label.config(text=self.get_importance_description(data.get('importance', 'normal')))

        # تحديث الوصف
        self.description_text.delete(1.0, tk.END)
        description = data.get('description', 'لا يوجد وصف متاح')
        
        # إضافة معلومات إضافية حسب النوع
        if data.get('innovation'):
            description += f"\n\n🌟 مستوى الابتكار: {self.get_innovation_description(data['innovation'])}"
        
        if data.get('type') == 'math_file':
            description += "\n\n🧮 هذا ملف يحتوي على نظام رياضي ثوري من إبداع باسل يحيى عبدالله"
        
        self.description_text.insert(tk.END, fix_arabic_text(description))

        # تحديث العلاقات
        self.relations_text.delete(1.0, tk.END)
        relations = self.get_relations_info(clean_name, data)
        self.relations_text.insert(tk.END, fix_arabic_text(relations))

        # تحديث شريط الحالة
        self.status_var.set(fix_label_text(f"عرض تفاصيل: {clean_name}"))

    def get_type_description(self, item_type):
        """وصف نوع العنصر"""
        descriptions = {
            'root': 'جذر المشروع',
            'core_folder': 'مجلد أساسي',
            'core_file': 'ملف أساسي',
            'interfaces_folder': 'مجلد الواجهات',
            'interface_folder': 'مجلد واجهة',
            'math_folder': 'مجلد رياضي',
            'math_file': 'ملف رياضي',
            'utility_file': 'ملف مساعد',
            'launcher_file': 'ملف تشغيل',
            'entry_file': 'نقطة دخول',
            'setup_file': 'ملف إعداد',
            'legacy_folder': 'مجلد تراثي',
            'legacy_file': 'ملف تراثي',
            'reference_folder': 'مجلد مرجعي'
        }
        return descriptions.get(item_type, 'غير محدد')

    def get_importance_description(self, importance):
        """وصف مستوى الأهمية"""
        descriptions = {
            'critical': '🔴 حرج - أساسي للنظام',
            'high': '🟡 عالي - مهم جداً',
            'medium': '🟢 متوسط - مفيد',
            'low': '🔵 منخفض - اختياري',
            'normal': '⚪ عادي'
        }
        return descriptions.get(importance, '⚪ عادي')

    def get_innovation_description(self, innovation):
        """وصف مستوى الابتكار"""
        descriptions = {
            'revolutionary': '🌟 ثوري - اكتشاف جديد كلياً',
            'advanced': '💡 متقدم - تطوير مبتكر',
            'standard': '📋 قياسي - تطبيق عادي'
        }
        return descriptions.get(innovation, '📋 قياسي')

    def get_relations_info(self, name, data):
        """معلومات العلاقات والتبعيات"""
        relations = "🔗 العلاقات والتبعيات:\n\n"
        
        if 'basira_simple_demo.py' in name:
            relations += """
🎯 النواة الرئيسية للنظام:
• يستورد جميع الأنظمة الرياضية من core/
• تستورده جميع الواجهات
• نقطة التكامل المركزية

📥 يستورد من:
• general_shape_equation.py
• innovative_calculus.py  
• revolutionary_decomposition.py
• expert_explorer_system.py

📤 يُستورد بواسطة:
• جميع ملفات test_*_interface.py
• run_all_interfaces.py
• START_BASIRA_SYSTEM.py
            """
        elif 'interfaces' in name:
            relations += """
🖥️ الواجهات التفاعلية:
• تستورد من basira_simple_demo.py
• تستخدم arabic_text_handler.py
• تعمل بشكل مستقل أو موحد

🔄 التفاعل:
• المستخدم → الواجهة → النواة → الأنظمة الرياضية
• النتائج ← معالج النصوص ← النواة ← الأنظمة
            """
        elif 'core' in name:
            relations += """
🧮 الأنظمة الرياضية الثورية:
• تُستورد بواسطة basira_simple_demo.py
• تعمل بشكل متكامل ومترابط
• تحتوي على إبداعات باسل يحيى عبدالله

🔬 التكامل:
• المعادلة العامة → أساس جميع الأنظمة
• النظام المبتكر → تكامل = V × A
• التفكيك الثوري → A = x.dA - ∫x.d2A
• نظام الخبير → توجيه وإرشاد
            """
        elif 'arabic_text_handler.py' in name:
            relations += """
🔤 معالج النصوص العربية:
• يُستورد بواسطة جميع الواجهات
• يحل مشكلة اتجاه النصوص العربية
• يدعم النصوص المختلطة

📚 المكتبات المطلوبة:
• arabic-reshaper
• python-bidi

🔧 الاستخدام:
• fix_arabic_text() - إصلاح عام
• fix_button_text() - نصوص الأزرار
• fix_title_text() - العناوين
            """
        else:
            relations += """
📋 معلومات عامة:
• جزء من هيكلية نظام بصيرة
• يساهم في الوظائف الإجمالية للنظام
• مترابط مع المكونات الأخرى

🔍 للمزيد من التفاصيل:
• راجع دليل هيكلية المشروع
• اقرأ التوثيق الداخلي للملف
• جرب تشغيل النظام لفهم التفاعل
            """
        
        return relations

    def run(self):
        """تشغيل المصور"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    print("🗺️ بدء تشغيل مصور هيكلية نظام بصيرة...")
    
    try:
        visualizer = ProjectStructureVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل المصور: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
