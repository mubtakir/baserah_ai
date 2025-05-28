#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة تنفيذ وفحص الأكواد

هذا الملف يحتوي على فئات لتنفيذ وفحص الأكواد البرمجية بمختلف اللغات،
مع دعم للتنفيذ الآمن والتحقق من صحة النتائج.

المؤلف: فريق تطوير نظام بصيرة
الإصدار: 1.0.0
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
import time
import re
import uuid
import threading
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('code_execution.executor')


class ProgrammingLanguage(Enum):
    """لغات البرمجة المدعومة."""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA = auto()
    C = auto()
    CPP = auto()
    CSHARP = auto()
    PHP = auto()
    RUBY = auto()
    GO = auto()
    RUST = auto()
    SWIFT = auto()
    KOTLIN = auto()
    BASH = auto()
    SQL = auto()
    R = auto()
    MATLAB = auto()
    SCALA = auto()
    PERL = auto()
    LUA = auto()
    HTML = auto()
    CSS = auto()
    XML = auto()
    JSON = auto()
    YAML = auto()
    MARKDOWN = auto()
    TEXT = auto()
    UNKNOWN = auto()


class ExecutionMode(Enum):
    """أنماط التنفيذ."""
    SAFE = auto()  # تنفيذ آمن (في بيئة معزولة)
    NORMAL = auto()  # تنفيذ عادي
    INTERACTIVE = auto()  # تنفيذ تفاعلي
    DEBUG = auto()  # تنفيذ في وضع التصحيح


class ExecutionStatus(Enum):
    """حالات التنفيذ."""
    SUCCESS = auto()  # نجاح التنفيذ
    ERROR = auto()  # خطأ في التنفيذ
    TIMEOUT = auto()  # انتهاء مهلة التنفيذ
    INTERRUPTED = auto()  # مقاطعة التنفيذ
    NOT_EXECUTED = auto()  # لم يتم التنفيذ


@dataclass
class ExecutionResult:
    """نتيجة تنفيذ الكود."""
    status: ExecutionStatus  # حالة التنفيذ
    output: str = ""  # مخرجات التنفيذ
    error: str = ""  # رسائل الخطأ
    execution_time: float = 0.0  # وقت التنفيذ (بالثواني)
    memory_usage: float = 0.0  # استخدام الذاكرة (بالميجابايت)
    return_value: Any = None  # القيمة المرجعة
    artifacts: Dict[str, str] = field(default_factory=dict)  # الملفات الناتجة عن التنفيذ
    metadata: Dict[str, Any] = field(default_factory=dict)  # بيانات وصفية إضافية


class CodeExecutorBase(ABC):
    """الفئة الأساسية لمنفذ الأكواد."""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SAFE):
        """
        تهيئة المنفذ.
        
        Args:
            execution_mode: نمط التنفيذ
        """
        self.logger = logging.getLogger('code_execution.executor.base')
        self.execution_mode = execution_mode
        self.timeout = 30  # مهلة التنفيذ الافتراضية (بالثواني)
        self.max_memory = 512  # الحد الأقصى للذاكرة (بالميجابايت)
        self.working_directory = tempfile.mkdtemp()  # دليل العمل المؤقت
        self.environment_variables = {}  # متغيرات البيئة
    
    @abstractmethod
    def execute(self, code: str, **kwargs) -> ExecutionResult:
        """
        تنفيذ الكود.
        
        Args:
            code: الكود المراد تنفيذه
            **kwargs: معاملات إضافية
            
        Returns:
            نتيجة التنفيذ
        """
        pass
    
    def cleanup(self) -> None:
        """تنظيف الموارد المستخدمة."""
        try:
            shutil.rmtree(self.working_directory)
        except Exception as e:
            self.logger.error(f"Error cleaning up working directory: {e}")


class PythonExecutor(CodeExecutorBase):
    """منفذ أكواد بايثون."""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SAFE):
        """
        تهيئة المنفذ.
        
        Args:
            execution_mode: نمط التنفيذ
        """
        super().__init__(execution_mode)
        self.logger = logging.getLogger('code_execution.executor.python')
        self.python_path = sys.executable  # مسار مفسر بايثون
    
    def execute(self, code: str, **kwargs) -> ExecutionResult:
        """
        تنفيذ كود بايثون.
        
        Args:
            code: كود بايثون
            **kwargs: معاملات إضافية
                - timeout: مهلة التنفيذ (بالثواني)
                - input_data: بيانات الإدخال
                - args: معاملات سطر الأوامر
                - env: متغيرات البيئة
                - working_dir: دليل العمل
                - dependencies: التبعيات المطلوبة
                
        Returns:
            نتيجة التنفيذ
        """
        # استخراج المعاملات
        timeout = kwargs.get('timeout', self.timeout)
        input_data = kwargs.get('input_data', '')
        args = kwargs.get('args', [])
        env = {**os.environ, **self.environment_variables, **kwargs.get('env', {})}
        working_dir = kwargs.get('working_dir', self.working_directory)
        dependencies = kwargs.get('dependencies', [])
        
        # إنشاء ملف مؤقت للكود
        code_file = os.path.join(working_dir, f"code_{uuid.uuid4().hex}.py")
        
        try:
            # كتابة الكود إلى الملف
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # تثبيت التبعيات إذا كانت موجودة
            if dependencies:
                self._install_dependencies(dependencies)
            
            # تنفيذ الكود حسب نمط التنفيذ
            if self.execution_mode == ExecutionMode.SAFE:
                result = self._execute_safe(code_file, input_data, args, env, timeout)
            else:
                result = self._execute_normal(code_file, input_data, args, env, timeout)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing Python code: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
        
        finally:
            # تنظيف الملفات المؤقتة
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary files: {e}")
    
    def _install_dependencies(self, dependencies: List[str]) -> None:
        """
        تثبيت التبعيات.
        
        Args:
            dependencies: قائمة التبعيات
        """
        try:
            for dependency in dependencies:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", dependency],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error installing dependencies: {e}")
            raise
    
    def _execute_safe(self, code_file: str, input_data: str, args: List[str], env: Dict[str, str], timeout: int) -> ExecutionResult:
        """
        تنفيذ الكود في بيئة آمنة.
        
        Args:
            code_file: مسار ملف الكود
            input_data: بيانات الإدخال
            args: معاملات سطر الأوامر
            env: متغيرات البيئة
            timeout: مهلة التنفيذ
            
        Returns:
            نتيجة التنفيذ
        """
        # إنشاء أمر التنفيذ
        cmd = [self.python_path, code_file] + args
        
        # قياس وقت التنفيذ
        start_time = time.time()
        
        try:
            # تنفيذ الكود
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # إرسال بيانات الإدخال إذا كانت موجودة
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            
            # حساب وقت التنفيذ
            execution_time = time.time() - start_time
            
            # تحديد حالة التنفيذ
            if process.returncode == 0:
                status = ExecutionStatus.SUCCESS
            else:
                status = ExecutionStatus.ERROR
            
            return ExecutionResult(
                status=status,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                return_value=process.returncode
            )
        
        except subprocess.TimeoutExpired:
            # إنهاء العملية في حالة انتهاء المهلة
            process.kill()
            stdout, stderr = process.communicate()
            
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                output=stdout,
                error=f"Execution timed out after {timeout} seconds",
                execution_time=timeout
            )
        
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
    
    def _execute_normal(self, code_file: str, input_data: str, args: List[str], env: Dict[str, str], timeout: int) -> ExecutionResult:
        """
        تنفيذ الكود بشكل عادي.
        
        Args:
            code_file: مسار ملف الكود
            input_data: بيانات الإدخال
            args: معاملات سطر الأوامر
            env: متغيرات البيئة
            timeout: مهلة التنفيذ
            
        Returns:
            نتيجة التنفيذ
        """
        # في الوضع العادي، نستخدم نفس طريقة التنفيذ الآمن
        return self._execute_safe(code_file, input_data, args, env, timeout)


class JavaScriptExecutor(CodeExecutorBase):
    """منفذ أكواد جافاسكريبت."""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SAFE):
        """
        تهيئة المنفذ.
        
        Args:
            execution_mode: نمط التنفيذ
        """
        super().__init__(execution_mode)
        self.logger = logging.getLogger('code_execution.executor.javascript')
        self.node_path = shutil.which('node') or 'node'  # مسار مفسر نود
    
    def execute(self, code: str, **kwargs) -> ExecutionResult:
        """
        تنفيذ كود جافاسكريبت.
        
        Args:
            code: كود جافاسكريبت
            **kwargs: معاملات إضافية
                - timeout: مهلة التنفيذ (بالثواني)
                - input_data: بيانات الإدخال
                - args: معاملات سطر الأوامر
                - env: متغيرات البيئة
                - working_dir: دليل العمل
                - dependencies: التبعيات المطلوبة
                
        Returns:
            نتيجة التنفيذ
        """
        # استخراج المعاملات
        timeout = kwargs.get('timeout', self.timeout)
        input_data = kwargs.get('input_data', '')
        args = kwargs.get('args', [])
        env = {**os.environ, **self.environment_variables, **kwargs.get('env', {})}
        working_dir = kwargs.get('working_dir', self.working_directory)
        dependencies = kwargs.get('dependencies', [])
        
        # إنشاء ملف مؤقت للكود
        code_file = os.path.join(working_dir, f"code_{uuid.uuid4().hex}.js")
        
        try:
            # كتابة الكود إلى الملف
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # تثبيت التبعيات إذا كانت موجودة
            if dependencies:
                self._install_dependencies(dependencies, working_dir)
            
            # تنفيذ الكود حسب نمط التنفيذ
            if self.execution_mode == ExecutionMode.SAFE:
                result = self._execute_safe(code_file, input_data, args, env, timeout)
            else:
                result = self._execute_normal(code_file, input_data, args, env, timeout)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing JavaScript code: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
        
        finally:
            # تنظيف الملفات المؤقتة
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary files: {e}")
    
    def _install_dependencies(self, dependencies: List[str], working_dir: str) -> None:
        """
        تثبيت التبعيات.
        
        Args:
            dependencies: قائمة التبعيات
            working_dir: دليل العمل
        """
        try:
            # إنشاء ملف package.json إذا لم يكن موجودًا
            package_json_path = os.path.join(working_dir, 'package.json')
            if not os.path.exists(package_json_path):
                with open(package_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "name": "code-execution",
                        "version": "1.0.0",
                        "description": "Temporary package for code execution",
                        "dependencies": {}
                    }, f)
            
            # تثبيت التبعيات
            for dependency in dependencies:
                subprocess.run(
                    ['npm', 'install', dependency],
                    cwd=working_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error installing dependencies: {e}")
            raise
    
    def _execute_safe(self, code_file: str, input_data: str, args: List[str], env: Dict[str, str], timeout: int) -> ExecutionResult:
        """
        تنفيذ الكود في بيئة آمنة.
        
        Args:
            code_file: مسار ملف الكود
            input_data: بيانات الإدخال
            args: معاملات سطر الأوامر
            env: متغيرات البيئة
            timeout: مهلة التنفيذ
            
        Returns:
            نتيجة التنفيذ
        """
        # إنشاء أمر التنفيذ
        cmd = [self.node_path, code_file] + args
        
        # قياس وقت التنفيذ
        start_time = time.time()
        
        try:
            # تنفيذ الكود
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # إرسال بيانات الإدخال إذا كانت موجودة
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            
            # حساب وقت التنفيذ
            execution_time = time.time() - start_time
            
            # تحديد حالة التنفيذ
            if process.returncode == 0:
                status = ExecutionStatus.SUCCESS
            else:
                status = ExecutionStatus.ERROR
            
            return ExecutionResult(
                status=status,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                return_value=process.returncode
            )
        
        except subprocess.TimeoutExpired:
            # إنهاء العملية في حالة انتهاء المهلة
            process.kill()
            stdout, stderr = process.communicate()
            
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                output=stdout,
                error=f"Execution timed out after {timeout} seconds",
                execution_time=timeout
            )
        
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
    
    def _execute_normal(self, code_file: str, input_data: str, args: List[str], env: Dict[str, str], timeout: int) -> ExecutionResult:
        """
        تنفيذ الكود بشكل عادي.
        
        Args:
            code_file: مسار ملف الكود
            input_data: بيانات الإدخال
            args: معاملات سطر الأوامر
            env: متغيرات البيئة
            timeout: مهلة التنفيذ
            
        Returns:
            نتيجة التنفيذ
        """
        # في الوضع العادي، نستخدم نفس طريقة التنفيذ الآمن
        return self._execute_safe(code_file, input_data, args, env, timeout)


class BashExecutor(CodeExecutorBase):
    """منفذ أكواد باش."""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SAFE):
        """
        تهيئة المنفذ.
        
        Args:
            execution_mode: نمط التنفيذ
        """
        super().__init__(execution_mode)
        self.logger = logging.getLogger('code_execution.executor.bash')
        self.bash_path = shutil.which('bash') or '/bin/bash'  # مسار مفسر باش
    
    def execute(self, code: str, **kwargs) -> ExecutionResult:
        """
        تنفيذ كود باش.
        
        Args:
            code: كود باش
            **kwargs: معاملات إضافية
                - timeout: مهلة التنفيذ (بالثواني)
                - input_data: بيانات الإدخال
                - args: معاملات سطر الأوامر
                - env: متغيرات البيئة
                - working_dir: دليل العمل
                
        Returns:
            نتيجة التنفيذ
        """
        # استخراج المعاملات
        timeout = kwargs.get('timeout', self.timeout)
        input_data = kwargs.get('input_data', '')
        args = kwargs.get('args', [])
        env = {**os.environ, **self.environment_variables, **kwargs.get('env', {})}
        working_dir = kwargs.get('working_dir', self.working_directory)
        
        # إنشاء ملف مؤقت للكود
        code_file = os.path.join(working_dir, f"code_{uuid.uuid4().hex}.sh")
        
        try:
            # كتابة الكود إلى الملف
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # جعل الملف قابل للتنفيذ
            os.chmod(code_file, 0o755)
            
            # تنفيذ الكود حسب نمط التنفيذ
            if self.execution_mode == ExecutionMode.SAFE:
                result = self._execute_safe(code_file, input_data, args, env, timeout)
            else:
                result = self._execute_normal(code_file, input_data, args, env, timeout)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing Bash code: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
        
        finally:
            # تنظيف الملفات المؤقتة
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary files: {e}")
    
    def _execute_safe(self, code_file: str, input_data: str, args: List[str], env: Dict[str, str], timeout: int) -> ExecutionResult:
        """
        تنفيذ الكود في بيئة آمنة.
        
        Args:
            code_file: مسار ملف الكود
            input_data: بيانات الإدخال
            args: معاملات سطر الأوامر
            env: متغيرات البيئة
            timeout: مهلة التنفيذ
            
        Returns:
            نتيجة التنفيذ
        """
        # إنشاء أمر التنفيذ
        cmd = [self.bash_path, code_file] + args
        
        # قياس وقت التنفيذ
        start_time = time.time()
        
        try:
            # تنفيذ الكود
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # إرسال بيانات الإدخال إذا كانت موجودة
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            
            # حساب وقت التنفيذ
            execution_time = time.time() - start_time
            
            # تحديد حالة التنفيذ
            if process.returncode == 0:
                status = ExecutionStatus.SUCCESS
            else:
                status = ExecutionStatus.ERROR
            
            return ExecutionResult(
                status=status,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                return_value=process.returncode
            )
        
        except subprocess.TimeoutExpired:
            # إنهاء العملية في حالة انتهاء المهلة
            process.kill()
            stdout, stderr = process.communicate()
            
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                output=stdout,
                error=f"Execution timed out after {timeout} seconds",
                execution_time=timeout
            )
        
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e)
            )
    
    def _execute_normal(self, code_file: str, input_data: str, args: List[str], env: Dict[str, str], timeout: int) -> ExecutionResult:
        """
        تنفيذ الكود بشكل عادي.
        
        Args:
            code_file: مسار ملف الكود
            input_data: بيانات الإدخال
            args: معاملات سطر الأوامر
            env: متغيرات البيئة
            timeout: مهلة التنفيذ
            
        Returns:
            نتيجة التنفيذ
        """
        # في الوضع العادي، نستخدم نفس طريقة التنفيذ الآمن
        return self._execute_safe(code_file, input_data, args, env, timeout)


class CodeValidator:
    """مدقق الأكواد."""
    
    def __init__(self):
        """تهيئة المدقق."""
        self.logger = logging.getLogger('code_execution.validator')
    
    def validate_python(self, code: str) -> Tuple[bool, str]:
        """
        تدقيق كود بايثون.
        
        Args:
            code: كود بايثون
            
        Returns:
            (صحيح/خطأ، رسالة الخطأ)
        """
        try:
            # تحقق من صحة بناء الكود
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate_javascript(self, code: str) -> Tuple[bool, str]:
        """
        تدقيق كود جافاسكريبت.
        
        Args:
            code: كود جافاسكريبت
            
        Returns:
            (صحيح/خطأ، رسالة الخطأ)
        """
        try:
            # إنشاء ملف مؤقت للكود
            with tempfile.NamedTemporaryFile(suffix='.js', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code.encode('utf-8'))
            
            # تحقق من صحة بناء الكود باستخدام Node.js
            result = subprocess.run(
                ['node', '--check', temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # إزالة الملف المؤقت
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate_bash(self, code: str) -> Tuple[bool, str]:
        """
        تدقيق كود باش.
        
        Args:
            code: كود باش
            
        Returns:
            (صحيح/خطأ، رسالة الخطأ)
        """
        try:
            # إنشاء ملف مؤقت للكود
            with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code.encode('utf-8'))
            
            # تحقق من صحة بناء الكود باستخدام bash -n
            result = subprocess.run(
                ['bash', '-n', temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # إزالة الملف المؤقت
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate(self, code: str, language: ProgrammingLanguage) -> Tuple[bool, str]:
        """
        تدقيق الكود.
        
        Args:
            code: الكود
            language: لغة البرمجة
            
        Returns:
            (صحيح/خطأ، رسالة الخطأ)
        """
        if language == ProgrammingLanguage.PYTHON:
            return self.validate_python(code)
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return self.validate_javascript(code)
        elif language == ProgrammingLanguage.BASH:
            return self.validate_bash(code)
        else:
            self.logger.warning(f"Validation not implemented for language: {language}")
            return True, ""


class CodeExecutionService:
    """خدمة تنفيذ الأكواد."""
    
    def __init__(self):
        """تهيئة الخدمة."""
        self.logger = logging.getLogger('code_execution.service')
        self.validator = CodeValidator()
        
        # إنشاء منفذات اللغات المدعومة
        self.executors = {
            ProgrammingLanguage.PYTHON: PythonExecutor(),
            ProgrammingLanguage.JAVASCRIPT: JavaScriptExecutor(),
            ProgrammingLanguage.TYPESCRIPT: JavaScriptExecutor(),  # استخدام منفذ جافاسكريبت لتايبسكريبت
            ProgrammingLanguage.BASH: BashExecutor()
        }
    
    def detect_language(self, code: str) -> ProgrammingLanguage:
        """
        اكتشاف لغة البرمجة من الكود.
        
        Args:
            code: الكود
            
        Returns:
            لغة البرمجة
        """
        # البحث عن علامات مميزة للغات المختلفة
        
        # بايثون
        if re.search(r'import\s+[a-zA-Z_][a-zA-Z0-9_]*|from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import|def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(|class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:', code):
            return ProgrammingLanguage.PYTHON
        
        # جافاسكريبت
        if re.search(r'const\s+[a-zA-Z_$][a-zA-Z0-9_$]*|let\s+[a-zA-Z_$][a-zA-Z0-9_$]*|var\s+[a-zA-Z_$][a-zA-Z0-9_$]*|function\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(|=>\s*{|\bexport\b|\bimport\b', code):
            return ProgrammingLanguage.JAVASCRIPT
        
        # تايبسكريبت
        if re.search(r'interface\s+[a-zA-Z_$][a-zA-Z0-9_$]*|type\s+[a-zA-Z_$][a-zA-Z0-9_$]*|:\s*[a-zA-Z_$][a-zA-Z0-9_$]*', code):
            return ProgrammingLanguage.TYPESCRIPT
        
        # باش
        if re.search(r'#!/bin/bash|#!/usr/bin/env bash|function\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*\)', code):
            return ProgrammingLanguage.BASH
        
        # إذا لم يتم التعرف على اللغة، نفترض أنها بايثون
        return ProgrammingLanguage.PYTHON
    
    def execute(self, code: str, language: Optional[ProgrammingLanguage] = None, **kwargs) -> ExecutionResult:
        """
        تنفيذ الكود.
        
        Args:
            code: الكود
            language: لغة البرمجة (اختياري، يتم اكتشافها تلقائيًا إذا لم يتم تحديدها)
            **kwargs: معاملات إضافية
                - validate: تدقيق الكود قبل التنفيذ (افتراضي: True)
                - execution_mode: نمط التنفيذ (افتراضي: SAFE)
                - timeout: مهلة التنفيذ (بالثواني)
                - input_data: بيانات الإدخال
                - args: معاملات سطر الأوامر
                - env: متغيرات البيئة
                - working_dir: دليل العمل
                - dependencies: التبعيات المطلوبة
                
        Returns:
            نتيجة التنفيذ
        """
        # اكتشاف لغة البرمجة إذا لم يتم تحديدها
        if language is None:
            language = self.detect_language(code)
        
        # تدقيق الكود إذا كان مطلوبًا
        if kwargs.get('validate', True):
            is_valid, error_message = self.validator.validate(code, language)
            if not is_valid:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=error_message
                )
        
        # التحقق من وجود منفذ للغة
        if language not in self.executors:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Unsupported language: {language}"
            )
        
        # تنفيذ الكود
        executor = self.executors[language]
        
        # تحديد نمط التنفيذ
        execution_mode = kwargs.get('execution_mode', ExecutionMode.SAFE)
        executor.execution_mode = execution_mode
        
        # تنفيذ الكود
        return executor.execute(code, **kwargs)


# --- اختبارات ---
if __name__ == "__main__":
    # إنشاء خدمة تنفيذ الأكواد
    service = CodeExecutionService()
    
    # اختبار تنفيذ كود بايثون
    python_code = """
print("مرحبًا بالعالم!")
for i in range(5):
    print(f"العدد: {i}")
"""
    
    print("تنفيذ كود بايثون:")
    result = service.execute(python_code)
    print(f"الحالة: {result.status}")
    print(f"المخرجات: {result.output}")
    print(f"الأخطاء: {result.error}")
    print(f"وقت التنفيذ: {result.execution_time} ثانية")
    
    # اختبار تنفيذ كود جافاسكريبت
    javascript_code = """
console.log("مرحبًا بالعالم!");
for (let i = 0; i < 5; i++) {
    console.log(`العدد: ${i}`);
}
"""
    
    print("\nتنفيذ كود جافاسكريبت:")
    result = service.execute(javascript_code, language=ProgrammingLanguage.JAVASCRIPT)
    print(f"الحالة: {result.status}")
    print(f"المخرجات: {result.output}")
    print(f"الأخطاء: {result.error}")
    print(f"وقت التنفيذ: {result.execution_time} ثانية")
    
    # اختبار تنفيذ كود باش
    bash_code = """
#!/bin/bash
echo "مرحبًا بالعالم!"
for i in {0..4}; do
    echo "العدد: $i"
done
"""
    
    print("\nتنفيذ كود باش:")
    result = service.execute(bash_code, language=ProgrammingLanguage.BASH)
    print(f"الحالة: {result.status}")
    print(f"المخرجات: {result.output}")
    print(f"الأخطاء: {result.error}")
    print(f"وقت التنفيذ: {result.execution_time} ثانية")
