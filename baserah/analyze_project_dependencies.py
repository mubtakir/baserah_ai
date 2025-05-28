#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Dependencies Analyzer for Basira System
محلل تبعيات المشروع لنظام بصيرة

Automatically analyzes and maps all dependencies in the project.
يحلل ويرسم جميع التبعيات في المشروع تلقائياً.

Author: Basira System Development Team
Created by: Basil Yahya Abdullah - Iraq/Mosul
Version: 1.0.0
"""

import os
import re
import ast
import sys
from datetime import datetime
from collections import defaultdict, deque

class ProjectDependencyAnalyzer:
    """محلل تبعيات المشروع"""
    
    def __init__(self, project_root="."):
        """تهيئة المحلل"""
        self.project_root = project_root
        self.dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        self.file_info = {}
        self.python_files = []
        
    def analyze_project(self):
        """تحليل المشروع بالكامل"""
        print("🔍 بدء تحليل تبعيات نظام بصيرة...")
        print("🔍 Starting Basira System dependency analysis...")
        
        # العثور على جميع ملفات Python
        self.find_python_files()
        
        # تحليل كل ملف
        for file_path in self.python_files:
            self.analyze_file(file_path)
        
        # إنشاء التقرير
        self.generate_report()
        
    def find_python_files(self):
        """العثور على جميع ملفات Python"""
        print("📁 البحث عن ملفات Python...")
        
        for root, dirs, files in os.walk(self.project_root):
            # تجاهل مجلدات معينة
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.python_files.append(file_path)
        
        print(f"✅ تم العثور على {len(self.python_files)} ملف Python")
        
    def analyze_file(self, file_path):
        """تحليل ملف واحد"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # معلومات أساسية عن الملف
            self.file_info[file_path] = {
                'size': len(content),
                'lines': len(content.splitlines()),
                'imports': [],
                'functions': [],
                'classes': [],
                'description': self.extract_description(content)
            }
            
            # تحليل الاستيرادات
            imports = self.extract_imports(content)
            self.file_info[file_path]['imports'] = imports
            
            # تحليل الدوال والكلاسات
            self.analyze_ast(file_path, content)
            
            # بناء خريطة التبعيات
            for imp in imports:
                if self.is_local_import(imp):
                    self.dependencies[file_path].add(imp)
                    self.reverse_dependencies[imp].add(file_path)
                    
        except Exception as e:
            print(f"⚠️ خطأ في تحليل {file_path}: {e}")
    
    def extract_description(self, content):
        """استخراج وصف الملف من docstring"""
        try:
            tree = ast.parse(content)
            if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant)):
                docstring = tree.body[0].value.value
                if isinstance(docstring, str):
                    # استخراج السطر الأول من الوصف
                    lines = docstring.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('"""') and not line.startswith("'''"):
                            return line
        except:
            pass
        return "لا يوجد وصف"
    
    def extract_imports(self, content):
        """استخراج جميع الاستيرادات"""
        imports = []
        
        # استيرادات import
        import_pattern = r'^import\s+([^\s#]+)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imports.append(match.group(1))
        
        # استيرادات from ... import
        from_pattern = r'^from\s+([^\s#]+)\s+import'
        for match in re.finditer(from_pattern, content, re.MULTILINE):
            imports.append(match.group(1))
        
        return imports
    
    def analyze_ast(self, file_path, content):
        """تحليل AST للدوال والكلاسات"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.file_info[file_path]['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    self.file_info[file_path]['classes'].append(node.name)
                    
        except Exception as e:
            print(f"⚠️ خطأ في تحليل AST لـ {file_path}: {e}")
    
    def is_local_import(self, import_name):
        """تحديد ما إذا كان الاستيراد محلي"""
        # قائمة المكتبات الخارجية المعروفة
        external_libs = {
            'tkinter', 'flask', 'numpy', 'matplotlib', 'PIL', 'datetime',
            'sys', 'os', 're', 'json', 'math', 'random', 'subprocess',
            'arabic_reshaper', 'bidi', 'ast', 'collections'
        }
        
        # فحص إذا كان الاستيراد محلي
        base_import = import_name.split('.')[0]
        return base_import not in external_libs and not base_import.startswith('_')
    
    def generate_report(self):
        """إنشاء تقرير شامل"""
        report_content = self.create_report_content()
        
        # حفظ التقرير
        report_file = "PROJECT_DEPENDENCIES_ANALYSIS.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ تم إنشاء تقرير التبعيات: {report_file}")
        
        # طباعة ملخص
        self.print_summary()
    
    def create_report_content(self):
        """إنشاء محتوى التقرير"""
        content = f"""# تحليل تبعيات نظام بصيرة
# Basira System Dependencies Analysis

## 📊 **ملخص التحليل**
## 📊 **Analysis Summary**

**📅 تاريخ التحليل:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**📁 عدد الملفات:** {len(self.python_files)}  
**🔗 إجمالي التبعيات:** {sum(len(deps) for deps in self.dependencies.values())}  

---

## 📁 **تفصيل الملفات**
## 📁 **File Details**

"""
        
        # تفصيل كل ملف
        for file_path in sorted(self.python_files):
            rel_path = os.path.relpath(file_path, self.project_root)
            info = self.file_info.get(file_path, {})
            
            content += f"""
### 📄 **{rel_path}**

**📝 الوصف:** {info.get('description', 'لا يوجد وصف')}  
**📏 الحجم:** {info.get('lines', 0)} سطر  
**🔧 الدوال:** {len(info.get('functions', []))}  
**🏗️ الكلاسات:** {len(info.get('classes', []))}  

#### 📥 **يستورد من:**
"""
            
            imports = info.get('imports', [])
            local_imports = [imp for imp in imports if self.is_local_import(imp)]
            
            if local_imports:
                for imp in local_imports:
                    content += f"- `{imp}`\n"
            else:
                content += "- لا توجد استيرادات محلية\n"
            
            content += "\n#### 📤 **يُستورد بواسطة:**\n"
            
            reverse_deps = self.reverse_dependencies.get(rel_path.replace('\\', '/'), set())
            if reverse_deps:
                for dep in sorted(reverse_deps):
                    content += f"- `{os.path.relpath(dep, self.project_root)}`\n"
            else:
                content += "- لا يُستورد بواسطة ملفات أخرى\n"
            
            # الدوال والكلاسات المهمة
            if info.get('functions'):
                content += f"\n#### ⚙️ **الدوال الرئيسية:**\n"
                for func in info['functions'][:5]:  # أول 5 دوال
                    content += f"- `{func}()`\n"
                if len(info['functions']) > 5:
                    content += f"- ... و {len(info['functions']) - 5} دوال أخرى\n"
            
            if info.get('classes'):
                content += f"\n#### 🏗️ **الكلاسات:**\n"
                for cls in info['classes']:
                    content += f"- `{cls}`\n"
            
            content += "\n---\n"
        
        # خريطة التبعيات
        content += self.create_dependency_map()
        
        # الملفات الأساسية
        content += self.create_core_files_analysis()
        
        return content
    
    def create_dependency_map(self):
        """إنشاء خريطة التبعيات"""
        content = """
## 🗺️ **خريطة التبعيات**
## 🗺️ **Dependency Map**

### 🔗 **التبعيات المباشرة:**

```
"""
        
        for file_path in sorted(self.dependencies.keys()):
            rel_path = os.path.relpath(file_path, self.project_root)
            deps = self.dependencies[file_path]
            
            if deps:
                content += f"{rel_path}\n"
                for dep in sorted(deps):
                    content += f"  ├── {dep}\n"
                content += "\n"
        
        content += "```\n\n"
        
        # الملفات الأكثر استيراداً
        content += "### 📊 **الملفات الأكثر استيراداً:**\n\n"
        
        import_counts = [(len(deps), file) for file, deps in self.reverse_dependencies.items()]
        import_counts.sort(reverse=True)
        
        for count, file in import_counts[:10]:
            if count > 0:
                content += f"- **{file}**: يُستورد بواسطة {count} ملف\n"
        
        return content
    
    def create_core_files_analysis(self):
        """تحليل الملفات الأساسية"""
        content = """
## 🎯 **تحليل الملفات الأساسية**
## 🎯 **Core Files Analysis**

### 🏗️ **الملفات الأساسية للنظام:**

"""
        
        core_files = [
            'basira_simple_demo.py',
            'arabic_text_handler.py', 
            'run_all_interfaces.py',
            'START_BASIRA_SYSTEM.py'
        ]
        
        for core_file in core_files:
            # البحث عن الملف في المشروع
            found_file = None
            for file_path in self.python_files:
                if core_file in file_path:
                    found_file = file_path
                    break
            
            if found_file:
                rel_path = os.path.relpath(found_file, self.project_root)
                info = self.file_info.get(found_file, {})
                deps_count = len(self.dependencies.get(found_file, set()))
                reverse_deps_count = len(self.reverse_dependencies.get(rel_path.replace('\\', '/'), set()))
                
                content += f"""
#### 📄 **{core_file}**
- **المسار:** `{rel_path}`
- **الوصف:** {info.get('description', 'لا يوجد وصف')}
- **الحجم:** {info.get('lines', 0)} سطر
- **يستورد من:** {deps_count} ملف
- **يُستورد بواسطة:** {reverse_deps_count} ملف
- **الأهمية:** {'🔴 حرج' if reverse_deps_count > 3 else '🟡 مهم' if reverse_deps_count > 1 else '🟢 عادي'}
"""
        
        content += """
### 🔄 **تدفق التبعيات الرئيسي:**

```
المستخدم
    ↓
START_BASIRA_SYSTEM.py
    ↓
run_all_interfaces.py أو basira_simple_demo.py
    ↓
الواجهات (desktop, web, hieroglyphic, brainstorm)
    ↓
arabic_text_handler.py (لمعالجة النصوص)
    ↓
الأنظمة الرياضية (core/)
```

### 💡 **توصيات للمطورين:**

1. **ابدأ بفهم `basira_simple_demo.py`** - هو قلب النظام
2. **ادرس `arabic_text_handler.py`** - لفهم معالجة النصوص
3. **استكشف مجلد `interfaces/`** - لفهم الواجهات المختلفة
4. **تعمق في `core/`** - للأنظمة الرياضية الثورية

### 🚨 **ملاحظات مهمة:**

- **لا تعدل الملفات الأساسية** بدون فهم كامل للتبعيات
- **اختبر دائماً** بعد أي تعديل على الملفات المترابطة
- **استخدم المصور التفاعلي** لفهم العلاقات بصرياً

---

*📅 تم إنشاء هذا التقرير تلقائياً في: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*🔍 محلل تبعيات نظام بصيرة - إبداع باسل يحيى عبدالله*
"""
        
        return content
    
    def print_summary(self):
        """طباعة ملخص التحليل"""
        print("\n" + "="*60)
        print("📊 ملخص تحليل تبعيات نظام بصيرة")
        print("="*60)
        print(f"📁 إجمالي الملفات: {len(self.python_files)}")
        print(f"🔗 إجمالي التبعيات: {sum(len(deps) for deps in self.dependencies.values())}")
        
        # الملفات الأكثر استيراداً
        import_counts = [(len(deps), file) for file, deps in self.reverse_dependencies.items()]
        import_counts.sort(reverse=True)
        
        print("\n🏆 الملفات الأكثر استيراداً:")
        for count, file in import_counts[:5]:
            if count > 0:
                print(f"  • {file}: {count} استيراد")
        
        # الملفات الأكثر تبعية
        dep_counts = [(len(deps), file) for file, deps in self.dependencies.items()]
        dep_counts.sort(reverse=True)
        
        print("\n📥 الملفات الأكثر تبعية:")
        for count, file in dep_counts[:5]:
            if count > 0:
                rel_path = os.path.relpath(file, self.project_root)
                print(f"  • {rel_path}: يستورد من {count} ملف")
        
        print("\n✅ تم إنجاز التحليل بنجاح!")
        print("📄 راجع ملف PROJECT_DEPENDENCIES_ANALYSIS.md للتفاصيل الكاملة")


def main():
    """الدالة الرئيسية"""
    print("🔍 محلل تبعيات نظام بصيرة")
    print("🔍 Basira System Dependencies Analyzer")
    print("="*50)
    
    try:
        analyzer = ProjectDependencyAnalyzer()
        analyzer.analyze_project()
        
        print("\n🎯 لعرض التحليل التفاعلي، شغل:")
        print("python3 baserah_system/project_structure_visualizer.py")
        
    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
