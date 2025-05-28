#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
فاحص العناصر التقليدية - Traditional Elements Scanner
يفحص النظام بالكامل للتأكد من عدم وجود أي عناصر تقليدية مخفية

Author: Basil Yahya Abdullah - Iraq/Mosul
"""

import os
import re
from typing import Dict, List, Set, Tuple
from pathlib import Path


class TraditionalElementsScanner:
    """فاحص العناصر التقليدية في النظام"""
    
    def __init__(self):
        """تهيئة الفاحص"""
        # العناصر التقليدية المحظورة
        self.forbidden_imports = {
            'torch', 'torch.nn', 'torch.optim', 'torch.nn.functional',
            'tensorflow', 'tf', 'keras', 'sklearn', 'scikit-learn',
            'pytorch', 'theano', 'caffe', 'mxnet', 'paddle',
            'gym', 'stable_baselines', 'ray', 'rllib'
        }
        
        self.forbidden_classes = {
            'nn.Module', 'nn.Linear', 'nn.Conv2d', 'nn.LSTM', 'nn.GRU',
            'nn.Sequential', 'nn.ReLU', 'nn.Sigmoid', 'nn.Tanh',
            'optim.Adam', 'optim.SGD', 'optim.RMSprop',
            'F.relu', 'F.sigmoid', 'F.softmax', 'F.mse_loss',
            'Model', 'Sequential', 'Dense', 'Conv2D', 'LSTM'
        }
        
        self.forbidden_concepts = {
            'neural_network', 'deep_learning', 'machine_learning',
            'reinforcement_learning', 'gradient_descent', 'backpropagation',
            'loss_function', 'optimizer', 'learning_rate', 'batch_size',
            'epoch', 'training', 'validation', 'test_set',
            'overfitting', 'underfitting', 'regularization',
            'dropout', 'batch_norm', 'activation_function'
        }
        
        self.forbidden_variables = {
            'learning_rate', 'lr', 'batch_size', 'epochs', 'loss',
            'optimizer', 'model', 'network', 'layers', 'weights',
            'biases', 'gradients', 'train_loader', 'test_loader',
            'accuracy', 'precision', 'recall', 'f1_score'
        }
        
        # النتائج
        self.scan_results = {
            'forbidden_imports': [],
            'forbidden_classes': [],
            'forbidden_concepts': [],
            'forbidden_variables': [],
            'suspicious_patterns': [],
            'clean_files': [],
            'problematic_files': []
        }
    
    def scan_file(self, file_path: str) -> Dict[str, List[str]]:
        """فحص ملف واحد"""
        file_issues = {
            'forbidden_imports': [],
            'forbidden_classes': [],
            'forbidden_concepts': [],
            'forbidden_variables': [],
            'suspicious_patterns': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # فحص الاستيرادات المحظورة
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower().strip()
                
                # فحص الاستيرادات
                if line_lower.startswith('import ') or line_lower.startswith('from '):
                    for forbidden in self.forbidden_imports:
                        if forbidden in line_lower:
                            file_issues['forbidden_imports'].append(
                                f"Line {line_num}: {line.strip()}"
                            )
                
                # فحص الفئات المحظورة
                for forbidden in self.forbidden_classes:
                    if forbidden in line:
                        file_issues['forbidden_classes'].append(
                            f"Line {line_num}: {line.strip()}"
                        )
                
                # فحص المفاهيم المحظورة
                for forbidden in self.forbidden_concepts:
                    if forbidden in line_lower:
                        file_issues['forbidden_concepts'].append(
                            f"Line {line_num}: {line.strip()}"
                        )
                
                # فحص المتغيرات المحظورة
                for forbidden in self.forbidden_variables:
                    # البحث عن المتغير كاسم منفصل
                    pattern = r'\b' + re.escape(forbidden) + r'\b'
                    if re.search(pattern, line_lower):
                        file_issues['forbidden_variables'].append(
                            f"Line {line_num}: {line.strip()}"
                        )
                
                # فحص الأنماط المشبوهة
                suspicious_patterns = [
                    r'\.backward\(\)', r'\.zero_grad\(\)', r'\.step\(\)',
                    r'\.train\(\)', r'\.eval\(\)', r'\.cuda\(\)',
                    r'torch\.', r'nn\.', r'F\.', r'optim\.',
                    r'loss\.', r'model\.', r'network\.'
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, line):
                        file_issues['suspicious_patterns'].append(
                            f"Line {line_num}: {line.strip()}"
                        )
        
        except Exception as e:
            file_issues['suspicious_patterns'].append(f"Error reading file: {e}")
        
        return file_issues
    
    def scan_directory(self, directory_path: str) -> Dict[str, Any]:
        """فحص مجلد كامل"""
        print(f"🔍 فحص المجلد: {directory_path}")
        
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # تجاهل مجلدات معينة
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        print(f"📁 تم العثور على {len(python_files)} ملف Python")
        
        # فحص كل ملف
        for file_path in python_files:
            print(f"   🔎 فحص: {os.path.basename(file_path)}")
            file_issues = self.scan_file(file_path)
            
            # تجميع النتائج
            has_issues = any(file_issues.values())
            
            if has_issues:
                self.scan_results['problematic_files'].append({
                    'file_path': file_path,
                    'issues': file_issues
                })
                
                # إضافة للقوائم العامة
                self.scan_results['forbidden_imports'].extend(
                    [f"{file_path}: {issue}" for issue in file_issues['forbidden_imports']]
                )
                self.scan_results['forbidden_classes'].extend(
                    [f"{file_path}: {issue}" for issue in file_issues['forbidden_classes']]
                )
                self.scan_results['forbidden_concepts'].extend(
                    [f"{file_path}: {issue}" for issue in file_issues['forbidden_concepts']]
                )
                self.scan_results['forbidden_variables'].extend(
                    [f"{file_path}: {issue}" for issue in file_issues['forbidden_variables']]
                )
                self.scan_results['suspicious_patterns'].extend(
                    [f"{file_path}: {issue}" for issue in file_issues['suspicious_patterns']]
                )
            else:
                self.scan_results['clean_files'].append(file_path)
        
        return self.scan_results
    
    def generate_report(self) -> str:
        """إنشاء تقرير شامل"""
        report = []
        report.append("🌟" + "="*80 + "🌟")
        report.append("🔍 تقرير فحص العناصر التقليدية في النظام")
        report.append("🌟" + "="*80 + "🌟")
        
        # إحصائيات عامة
        total_files = len(self.scan_results['clean_files']) + len(self.scan_results['problematic_files'])
        clean_files = len(self.scan_results['clean_files'])
        problematic_files = len(self.scan_results['problematic_files'])
        
        report.append(f"\n📊 الإحصائيات العامة:")
        report.append(f"   📁 إجمالي الملفات المفحوصة: {total_files}")
        report.append(f"   ✅ الملفات النظيفة: {clean_files}")
        report.append(f"   ⚠️ الملفات المشكوك فيها: {problematic_files}")
        report.append(f"   🎯 نسبة النظافة: {(clean_files/total_files)*100:.1f}%")
        
        # تفاصيل المشاكل
        if self.scan_results['forbidden_imports']:
            report.append(f"\n❌ الاستيرادات المحظورة ({len(self.scan_results['forbidden_imports'])}):")
            for issue in self.scan_results['forbidden_imports'][:10]:  # أول 10 فقط
                report.append(f"   {issue}")
            if len(self.scan_results['forbidden_imports']) > 10:
                report.append(f"   ... و {len(self.scan_results['forbidden_imports']) - 10} أخرى")
        
        if self.scan_results['forbidden_classes']:
            report.append(f"\n❌ الفئات المحظورة ({len(self.scan_results['forbidden_classes'])}):")
            for issue in self.scan_results['forbidden_classes'][:10]:
                report.append(f"   {issue}")
            if len(self.scan_results['forbidden_classes']) > 10:
                report.append(f"   ... و {len(self.scan_results['forbidden_classes']) - 10} أخرى")
        
        if self.scan_results['forbidden_concepts']:
            report.append(f"\n❌ المفاهيم المحظورة ({len(self.scan_results['forbidden_concepts'])}):")
            for issue in self.scan_results['forbidden_concepts'][:10]:
                report.append(f"   {issue}")
            if len(self.scan_results['forbidden_concepts']) > 10:
                report.append(f"   ... و {len(self.scan_results['forbidden_concepts']) - 10} أخرى")
        
        if self.scan_results['suspicious_patterns']:
            report.append(f"\n⚠️ الأنماط المشبوهة ({len(self.scan_results['suspicious_patterns'])}):")
            for issue in self.scan_results['suspicious_patterns'][:10]:
                report.append(f"   {issue}")
            if len(self.scan_results['suspicious_patterns']) > 10:
                report.append(f"   ... و {len(self.scan_results['suspicious_patterns']) - 10} أخرى")
        
        # الملفات المشكوك فيها
        if self.scan_results['problematic_files']:
            report.append(f"\n🚨 الملفات التي تحتاج إصلاح:")
            for file_info in self.scan_results['problematic_files'][:20]:
                file_path = file_info['file_path']
                issues_count = sum(len(issues) for issues in file_info['issues'].values())
                report.append(f"   📄 {os.path.basename(file_path)}: {issues_count} مشكلة")
        
        # التقييم النهائي
        report.append(f"\n🏆 التقييم النهائي:")
        if problematic_files == 0:
            report.append("   🎉 ممتاز! النظام خالٍ تماماً من العناصر التقليدية!")
        elif problematic_files <= total_files * 0.1:
            report.append("   ✅ جيد! معظم النظام نظيف مع بعض المشاكل البسيطة")
        elif problematic_files <= total_files * 0.3:
            report.append("   ⚠️ مقبول! يحتاج بعض التنظيف")
        else:
            report.append("   ❌ يحتاج عمل! الكثير من العناصر التقليدية موجودة")
        
        return "\n".join(report)
    
    def save_report(self, output_file: str):
        """حفظ التقرير في ملف"""
        report = self.generate_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 تم حفظ التقرير في: {output_file}")


def scan_baserah_system():
    """فحص نظام بصيرة بالكامل"""
    scanner = TraditionalElementsScanner()
    
    # فحص المجلدات الرئيسية
    directories_to_scan = [
        'baserah_system/learning',
        'baserah_system/mathematical_core',
        'baserah_system/revolutionary_core',
        'baserah_system/adaptive_mathematical_core',
        'baserah_system/symbolic_processing'
    ]
    
    print("🌟" + "="*80 + "🌟")
    print("🔍 بدء فحص نظام بصيرة للعناصر التقليدية")
    print("🌟" + "="*80 + "🌟")
    
    for directory in directories_to_scan:
        if os.path.exists(directory):
            scanner.scan_directory(directory)
        else:
            print(f"⚠️ المجلد غير موجود: {directory}")
    
    # إنشاء وحفظ التقرير
    report = scanner.generate_report()
    print(report)
    
    scanner.save_report('traditional_elements_scan_report.txt')
    
    return scanner.scan_results


if __name__ == "__main__":
    scan_results = scan_baserah_system()
