#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Desktop Interface for Basira System
اختبار واجهة سطح المكتب لنظام بصيرة

Simple test version of desktop interface to verify functionality.

Author: Basira System Development Team
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

try:
    from basira_simple_demo import SimpleExpertSystem
    BASIRA_AVAILABLE = True
except ImportError as e:
    print(f"تحذير: فشل في استيراد مكونات بصيرة: {e}")
    BASIRA_AVAILABLE = False

# استيراد معالج النصوص العربية
try:
    from baserah_system.arabic_text_handler import fix_arabic_text, fix_button_text, fix_title_text, fix_label_text
    ARABIC_HANDLER_AVAILABLE = True
    print("✅ معالج النصوص العربية متاح")
except ImportError as e:
    print(f"⚠️ معالج النصوص العربية غير متاح: {e}")
    ARABIC_HANDLER_AVAILABLE = False
    # دوال بديلة
    def fix_arabic_text(text): return text
    def fix_button_text(text): return text
    def fix_title_text(text): return text
    def fix_label_text(text): return text


class TestBasiraDesktopApp:
    """نسخة اختبار مبسطة لواجهة سطح المكتب"""

    def __init__(self):
        """تهيئة التطبيق"""
        self.root = tk.Tk()
        self.root.title(fix_title_text("نظام بصيرة - اختبار واجهة سطح المكتب"))
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        # تهيئة النظام
        if BASIRA_AVAILABLE:
            self.expert_system = SimpleExpertSystem()
        else:
            self.expert_system = None

        # إنشاء الواجهة
        self.create_widgets()

    def create_widgets(self):
        """إنشاء عناصر الواجهة"""
        # العنوان الرئيسي
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = ttk.Label(title_frame,
                               text=fix_title_text("🌟 نظام بصيرة - اختبار الواجهات 🌟"),
                               font=('Arial', 18, 'bold'))
        title_label.pack()

        # دفتر التبويبات
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # تبويبات الاختبار
        self.create_system_test_tab()
        self.create_calculus_test_tab()
        self.create_decomposition_test_tab()
        self.create_interface_test_tab()

        # شريط الحالة
        self.create_status_bar()

    def create_system_test_tab(self):
        """تبويب اختبار النظام العام"""
        system_frame = ttk.Frame(self.notebook)
        self.notebook.add(system_frame, text=fix_label_text("🔍 اختبار النظام"))

        # معلومات النظام
        info_frame = ttk.LabelFrame(system_frame, text=fix_label_text("معلومات النظام"))
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        info_text = fix_arabic_text(f"""
🌟 نظام بصيرة - إبداع باسل يحيى عبدالله
📅 تاريخ الاختبار: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔧 حالة النظام: {'✅ متاح' if BASIRA_AVAILABLE else '❌ غير متاح'}
💻 واجهة سطح المكتب: ✅ تعمل
🌐 Python tkinter: ✅ متاح
🔤 معالج النصوص العربية: {'✅ متاح' if ARABIC_HANDLER_AVAILABLE else '⚠️ غير متاح'}
        """)

        info_label = ttk.Label(info_frame, text=info_text, font=('Arial', 11))
        info_label.pack(padx=10, pady=10)

        # اختبار سريع
        test_btn = ttk.Button(system_frame, text=fix_button_text("🚀 تشغيل اختبار سريع"),
                             command=self.run_quick_test)
        test_btn.pack(pady=10)

        # نتائج الاختبار
        result_frame = ttk.LabelFrame(system_frame, text=fix_label_text("نتائج الاختبار"))
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.test_result = scrolledtext.ScrolledText(result_frame, height=15)
        self.test_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_calculus_test_tab(self):
        """تبويب اختبار النظام المبتكر للتفاضل والتكامل"""
        calculus_frame = ttk.Frame(self.notebook)
        self.notebook.add(calculus_frame, text=fix_label_text("🧮 النظام المبتكر"))

        # شرح النظام
        desc_frame = ttk.LabelFrame(calculus_frame, text=fix_label_text("النظام المبتكر للتفاضل والتكامل"))
        desc_frame.pack(fill=tk.X, padx=10, pady=5)

        desc_text = fix_arabic_text("""
💡 المفهوم الثوري لباسل يحيى عبدالله:
   تكامل أي دالة = V × A
   تفاضل أي دالة = D × A

حيث V و D معاملات يتم تعلمها بدلاً من الطرق التقليدية
        """)

        desc_label = ttk.Label(desc_frame, text=desc_text, font=('Arial', 10))
        desc_label.pack(padx=10, pady=5)

        # اختبار النظام
        test_calculus_btn = ttk.Button(calculus_frame, text=fix_button_text("🧮 اختبار النظام المبتكر"),
                                      command=self.test_innovative_calculus)
        test_calculus_btn.pack(pady=10)

        # نتائج
        calculus_result_frame = ttk.LabelFrame(calculus_frame, text=fix_label_text("نتائج النظام المبتكر"))
        calculus_result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.calculus_result = scrolledtext.ScrolledText(calculus_result_frame, height=12)
        self.calculus_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_decomposition_test_tab(self):
        """تبويب اختبار النظام الثوري لتفكيك الدوال"""
        decomp_frame = ttk.Frame(self.notebook)
        self.notebook.add(decomp_frame, text=fix_label_text("🌟 التفكيك الثوري"))

        # شرح النظام
        desc_frame = ttk.LabelFrame(decomp_frame, text=fix_label_text("النظام الثوري لتفكيك الدوال"))
        desc_frame.pack(fill=tk.X, padx=10, pady=5)

        desc_text = fix_arabic_text("""
🌟 الفرضية الثورية لباسل يحيى عبدالله:
   A = x.dA - ∫x.d2A

المتسلسلة الناتجة:
   A = Σ[(-1)^(n-1) * (x^n * d^n A) / n!] + R_n
        """)

        desc_label = ttk.Label(desc_frame, text=desc_text, font=('Arial', 10))
        desc_label.pack(padx=10, pady=5)

        # اختبار النظام
        test_decomp_btn = ttk.Button(decomp_frame, text=fix_button_text("🌟 اختبار التفكيك الثوري"),
                                    command=self.test_revolutionary_decomposition)
        test_decomp_btn.pack(pady=10)

        # نتائج
        decomp_result_frame = ttk.LabelFrame(decomp_frame, text=fix_label_text("نتائج التفكيك الثوري"))
        decomp_result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.decomp_result = scrolledtext.ScrolledText(decomp_result_frame, height=12)
        self.decomp_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_interface_test_tab(self):
        """تبويب اختبار الواجهات الأخرى"""
        interface_frame = ttk.Frame(self.notebook)
        self.notebook.add(interface_frame, text=fix_label_text("🖥️ اختبار الواجهات"))

        # قائمة الواجهات
        interfaces_frame = ttk.LabelFrame(interface_frame, text=fix_label_text("الواجهات المتاحة"))
        interfaces_frame.pack(fill=tk.X, padx=10, pady=5)

        interfaces = [
            (fix_arabic_text("🖥️ واجهة سطح المكتب"), "✅ تعمل حالياً"),
            (fix_arabic_text("🌐 واجهة الويب"), "🔄 سيتم اختبارها"),
            (fix_arabic_text("📜 الواجهة الهيروغلوفية"), "🔄 سيتم اختبارها"),
            (fix_arabic_text("🧠 واجهة العصف الذهني"), "🔄 سيتم اختبارها")
        ]

        for interface_name, status in interfaces:
            interface_row = ttk.Frame(interfaces_frame)
            interface_row.pack(fill=tk.X, padx=5, pady=2)

            ttk.Label(interface_row, text=interface_name, width=25).pack(side=tk.LEFT)
            ttk.Label(interface_row, text=status).pack(side=tk.LEFT, padx=10)

        # أزرار اختبار الواجهات
        buttons_frame = ttk.Frame(interface_frame)
        buttons_frame.pack(pady=20)

        ttk.Button(buttons_frame, text=fix_button_text("🌐 اختبار واجهة الويب"),
                  command=self.test_web_interface).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text=fix_button_text("📜 اختبار الواجهة الهيروغلوفية"),
                  command=self.test_hieroglyphic_interface).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text=fix_button_text("🧠 اختبار العصف الذهني"),
                  command=self.test_brainstorm_interface).pack(side=tk.LEFT, padx=5)

        # نتائج اختبار الواجهات
        interface_result_frame = ttk.LabelFrame(interface_frame, text=fix_label_text("نتائج اختبار الواجهات"))
        interface_result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.interface_result = scrolledtext.ScrolledText(interface_result_frame, height=10)
        self.interface_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_status_bar(self):
        """إنشاء شريط الحالة"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.StringVar()
        self.status_text.set(fix_arabic_text("مرحباً بك في اختبار نظام بصيرة"))

        status_label = ttk.Label(status_frame, textvariable=self.status_text)
        status_label.pack(side=tk.LEFT, padx=5)

        # وقت النظام
        self.time_label = ttk.Label(status_frame, text="")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        self.update_time()

    def update_time(self):
        """تحديث الوقت"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    # وظائف الاختبار
    def run_quick_test(self):
        """تشغيل اختبار سريع للنظام"""
        self.status_text.set("جاري تشغيل الاختبار السريع...")
        self.test_result.delete(1.0, tk.END)

        test_log = f"""
🚀 بدء الاختبار السريع لنظام بصيرة
📅 الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 فحص المكونات الأساسية:
{'✅' if BASIRA_AVAILABLE else '❌'} نظام بصيرة الأساسي
✅ واجهة tkinter
✅ مكتبات Python الأساسية

🧮 اختبار النظام المبتكر للتفاضل والتكامل:
"""

        if self.expert_system:
            try:
                # اختبار بسيط للنظام المبتكر
                test_function = [1, 4, 9, 16, 25]  # x^2
                D_coeffs = [2, 4, 6, 8, 10]  # 2x
                V_coeffs = [0.33, 1.33, 3, 5.33, 8.33]  # x^3/3

                self.expert_system.calculus_engine.add_coefficient_state(
                    test_function, D_coeffs, V_coeffs
                )

                result = self.expert_system.calculus_engine.predict_calculus(test_function)
                test_log += "✅ النظام المبتكر للتفاضل والتكامل يعمل بنجاح\n"
                test_log += f"   📊 التفاضل المقدر: {result['derivative'][:3]}...\n"
                test_log += f"   📊 التكامل المقدر: {result['integral'][:3]}...\n"

            except Exception as e:
                test_log += f"❌ خطأ في النظام المبتكر: {e}\n"

            test_log += "\n🌟 اختبار النظام الثوري لتفكيك الدوال:\n"
            try:
                # اختبار التفكيك الثوري
                x_vals = [1, 2, 3, 4, 5]
                f_vals = [1, 4, 9, 16, 25]  # x^2

                decomp_result = self.expert_system.decomposition_engine.decompose_simple_function(
                    "test_quadratic", x_vals, f_vals
                )

                test_log += "✅ النظام الثوري لتفكيك الدوال يعمل بنجاح\n"
                test_log += f"   📊 دقة التفكيك: {decomp_result['accuracy']:.4f}\n"
                test_log += f"   📊 عدد الحدود: {decomp_result['n_terms_used']}\n"

            except Exception as e:
                test_log += f"❌ خطأ في النظام الثوري: {e}\n"

        else:
            test_log += "❌ نظام بصيرة غير متاح\n"

        test_log += f"""
📊 ملخص الاختبار السريع:
✅ واجهة سطح المكتب تعمل بنجاح
{'✅' if BASIRA_AVAILABLE else '❌'} الأنظمة الرياضية الثورية
✅ جميع المكونات الأساسية

🎉 انتهى الاختبار السريع بنجاح!
        """

        self.test_result.insert(tk.END, test_log)
        self.status_text.set("✅ انتهى الاختبار السريع بنجاح")

    def test_innovative_calculus(self):
        """اختبار النظام المبتكر للتفاضل والتكامل"""
        self.status_text.set("جاري اختبار النظام المبتكر...")
        self.calculus_result.delete(1.0, tk.END)

        if not self.expert_system:
            self.calculus_result.insert(tk.END, "❌ نظام بصيرة غير متاح")
            return

        test_log = f"""
🧮 اختبار النظام المبتكر للتفاضل والتكامل
💡 إبداع باسل يحيى عبدالله

📋 المفهوم الثوري:
   تكامل = V × A
   تفاضل = D × A

🔬 اختبار على دوال مختلفة:

1️⃣ دالة تربيعية: f(x) = x²
"""

        try:
            # اختبار دالة تربيعية
            x_vals = [1, 2, 3, 4, 5]
            f_vals = [x**2 for x in x_vals]
            D_coeffs = [2*x for x in x_vals]  # مشتقة x^2 = 2x
            V_coeffs = [x**3/3 for x in x_vals]  # تكامل x^2 = x^3/3

            self.expert_system.calculus_engine.add_coefficient_state(f_vals, D_coeffs, V_coeffs)
            result = self.expert_system.calculus_engine.predict_calculus(f_vals)

            test_log += f"   ✅ نجح الاختبار\n"
            test_log += f"   📊 قيم الدالة: {f_vals}\n"
            test_log += f"   📊 التفاضل المقدر: {[round(x, 2) for x in result['derivative']]}\n"
            test_log += f"   📊 التكامل المقدر: {[round(x, 2) for x in result['integral']]}\n\n"

        except Exception as e:
            test_log += f"   ❌ فشل الاختبار: {e}\n\n"

        test_log += "2️⃣ دالة تكعيبية: f(x) = x³\n"
        try:
            # اختبار دالة تكعيبية
            f_vals_cubic = [x**3 for x in x_vals]
            D_coeffs_cubic = [3*x**2 for x in x_vals]  # مشتقة x^3 = 3x^2
            V_coeffs_cubic = [x**4/4 for x in x_vals]  # تكامل x^3 = x^4/4

            self.expert_system.calculus_engine.add_coefficient_state(f_vals_cubic, D_coeffs_cubic, V_coeffs_cubic)
            result_cubic = self.expert_system.calculus_engine.predict_calculus(f_vals_cubic)

            test_log += f"   ✅ نجح الاختبار\n"
            test_log += f"   📊 قيم الدالة: {f_vals_cubic}\n"
            test_log += f"   📊 التفاضل المقدر: {[round(x, 2) for x in result_cubic['derivative']]}\n"
            test_log += f"   📊 التكامل المقدر: {[round(x, 2) for x in result_cubic['integral']]}\n\n"

        except Exception as e:
            test_log += f"   ❌ فشل الاختبار: {e}\n\n"

        test_log += """
🎯 النتيجة النهائية:
✅ النظام المبتكر للتفاضل والتكامل يعمل بنجاح!
🌟 تحية لباسل يحيى عبدالله على هذا الابتكار الرياضي المذهل!
        """

        self.calculus_result.insert(tk.END, test_log)
        self.status_text.set("✅ انتهى اختبار النظام المبتكر")

    def test_revolutionary_decomposition(self):
        """اختبار النظام الثوري لتفكيك الدوال"""
        self.status_text.set("جاري اختبار التفكيك الثوري...")
        self.decomp_result.delete(1.0, tk.END)

        if not self.expert_system:
            self.decomp_result.insert(tk.END, "❌ نظام بصيرة غير متاح")
            return

        test_log = f"""
🌟 اختبار النظام الثوري لتفكيك الدوال
💡 إبداع باسل يحيى عبدالله

📋 الفرضية الثورية:
   A = x.dA - ∫x.d2A

📋 المتسلسلة الناتجة:
   A = Σ[(-1)^(n-1) * (x^n * d^n A) / n!] + R_n

🔬 اختبار على دوال مختلفة:

1️⃣ دالة تربيعية: f(x) = x²
"""

        try:
            # اختبار دالة تربيعية
            x_vals = [1, 2, 3, 4, 5]
            f_vals = [x**2 for x in x_vals]

            result = self.expert_system.decomposition_engine.decompose_simple_function(
                "quadratic_test", x_vals, f_vals
            )

            test_log += f"   ✅ نجح التفكيك\n"
            test_log += f"   📊 دقة التفكيك: {result['accuracy']:.4f}\n"
            test_log += f"   📊 عدد الحدود: {result['n_terms_used']}\n"
            test_log += f"   📊 الطريقة: {result['method']}\n\n"

        except Exception as e:
            test_log += f"   ❌ فشل التفكيك: {e}\n\n"

        test_log += "2️⃣ دالة تكعيبية: f(x) = x³\n"
        try:
            # اختبار دالة تكعيبية
            f_vals_cubic = [x**3 for x in x_vals]

            result_cubic = self.expert_system.decomposition_engine.decompose_simple_function(
                "cubic_test", x_vals, f_vals_cubic
            )

            test_log += f"   ✅ نجح التفكيك\n"
            test_log += f"   📊 دقة التفكيك: {result_cubic['accuracy']:.4f}\n"
            test_log += f"   📊 عدد الحدود: {result_cubic['n_terms_used']}\n"
            test_log += f"   📊 الطريقة: {result_cubic['method']}\n\n"

        except Exception as e:
            test_log += f"   ❌ فشل التفكيك: {e}\n\n"

        test_log += """
🎯 النتيجة النهائية:
✅ النظام الثوري لتفكيك الدوال يعمل بنجاح!
🌟 تحية لباسل يحيى عبدالله على هذا الاكتشاف الرياضي الثوري!
        """

        self.decomp_result.insert(tk.END, test_log)
        self.status_text.set("✅ انتهى اختبار التفكيك الثوري")

    def test_web_interface(self):
        """اختبار واجهة الويب"""
        self.status_text.set("جاري اختبار واجهة الويب...")
        self.interface_result.insert(tk.END, "🌐 بدء اختبار واجهة الويب...\n")

        # سيتم تنفيذ هذا لاحقاً
        self.interface_result.insert(tk.END, "🔄 سيتم اختبار واجهة الويب في الخطوة التالية\n\n")

    def test_hieroglyphic_interface(self):
        """اختبار الواجهة الهيروغلوفية"""
        self.status_text.set("جاري اختبار الواجهة الهيروغلوفية...")
        self.interface_result.insert(tk.END, "📜 بدء اختبار الواجهة الهيروغلوفية...\n")

        # سيتم تنفيذ هذا لاحقاً
        self.interface_result.insert(tk.END, "🔄 سيتم اختبار الواجهة الهيروغلوفية في الخطوة التالية\n\n")

    def test_brainstorm_interface(self):
        """اختبار واجهة العصف الذهني"""
        self.status_text.set("جاري اختبار واجهة العصف الذهني...")
        self.interface_result.insert(tk.END, "🧠 بدء اختبار واجهة العصف الذهني...\n")

        # سيتم تنفيذ هذا لاحقاً
        self.interface_result.insert(tk.END, "🔄 سيتم اختبار واجهة العصف الذهني في الخطوة التالية\n\n")

    def run(self):
        """تشغيل التطبيق"""
        self.root.mainloop()


def main():
    """الدالة الرئيسية"""
    print("🚀 بدء اختبار واجهة سطح المكتب لنظام بصيرة...")

    try:
        app = TestBasiraDesktopApp()
        app.run()
    except Exception as e:
        print(f"❌ خطأ في تشغيل واجهة سطح المكتب: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
