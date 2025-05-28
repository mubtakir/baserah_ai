#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Language Code Executor for Baserah System

This module implements a multi-language code executor that can execute code in
various programming languages and analyze the results using the General Shape Equation.

Author: Baserah System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import time
import uuid
import shutil
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback
import re
import threading
import signal
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import from core module
try:
    from core.general_shape_equation import (
        GeneralShapeEquation,
        EquationType,
        LearningMode,
        SymbolicExpression
    )
except ImportError:
    logging.warning("Failed to import from core.general_shape_equation, using placeholder implementation")
    
    # Placeholder implementations
    class EquationType(str, Enum):
        CODE = "code"
    
    class LearningMode(str, Enum):
        NONE = "none"
    
    class SymbolicExpression:
        def __init__(self, expression_str):
            self.expression_str = expression_str
    
    class GeneralShapeEquation:
        def __init__(self, equation_type=None, learning_mode=None):
            self.equation_type = equation_type or EquationType.CODE
            self.learning_mode = learning_mode or LearningMode.NONE
            self.components = {}
        
        def add_component(self, name, expression):
            self.components[name] = expression
        
        def evaluate(self, variable_values=None):
            return {"result": "Placeholder evaluation"}

# Configure logging
logger = logging.getLogger('code_execution.multi_language_executor')


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    TYPESCRIPT = "typescript"
    BASH = "bash"
    R = "r"
    JULIA = "julia"
    SCALA = "scala"
    PERL = "perl"
    HASKELL = "haskell"
    LUA = "lua"
    MATLAB = "matlab"


class ExecutionStatus(str, Enum):
    """Execution status codes."""
    SUCCESS = "success"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    PERMISSION_ERROR = "permission_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ExecutionParameters:
    """Parameters for code execution."""
    language: ProgrammingLanguage
    code: str
    input_data: Optional[str] = None
    timeout: float = 10.0  # Timeout in seconds
    memory_limit: Optional[int] = None  # Memory limit in MB
    additional_files: Dict[str, str] = field(default_factory=dict)  # {filename: content}
    command_line_args: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    execution_id: str
    language: ProgrammingLanguage
    status: ExecutionStatus
    stdout: str
    stderr: str
    execution_time: float
    memory_usage: Optional[float] = None  # Memory usage in MB
    return_code: Optional[int] = None
    compilation_output: Optional[str] = None
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class MultiLanguageExecutor:
    """
    Multi-Language Code Executor for Baserah System.
    
    This class implements a multi-language code executor that can execute code in
    various programming languages and analyze the results using the General Shape Equation.
    """
    
    def __init__(self):
        """Initialize the multi-language code executor."""
        # Initialize General Shape Equation
        self.equation = GeneralShapeEquation(
            equation_type=EquationType.CODE,
            learning_mode=LearningMode.NONE
        )
        
        # Initialize equation components
        self._initialize_equation_components()
        
        # Store execution history
        self.execution_history = {}
        
        # Create temporary directory for code execution
        self.temp_dir = tempfile.mkdtemp(prefix="baserah_executor_")
        
        # Register cleanup handler
        import atexit
        atexit.register(self._cleanup)
    
    def _initialize_equation_components(self) -> None:
        """Initialize the components of the General Shape Equation."""
        # Add components for code execution
        self.equation.add_component("code_quality", "analyze_code_quality(code)")
        self.equation.add_component("execution_success", "execution_status == 'success' ? 1.0 : 0.0")
        self.equation.add_component("execution_time_score", "1.0 / (1.0 + execution_time)")
        self.equation.add_component("memory_usage_score", "memory_limit > 0 ? 1.0 - (memory_usage / memory_limit) : 1.0")
        self.equation.add_component("overall_score", "execution_success * (0.4 * code_quality + 0.3 * execution_time_score + 0.3 * memory_usage_score)")
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")
    
    def execute_code(self, parameters: ExecutionParameters) -> ExecutionResult:
        """
        Execute code in the specified programming language.
        
        Args:
            parameters: Execution parameters
            
        Returns:
            Execution result
        """
        # Generate execution ID
        execution_id = f"exec_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Create execution directory
        execution_dir = os.path.join(self.temp_dir, execution_id)
        os.makedirs(execution_dir, exist_ok=True)
        
        # Set working directory
        working_dir = parameters.working_directory or execution_dir
        
        # Start execution timer
        start_time = time.time()
        
        try:
            # Execute code based on language
            if parameters.language == ProgrammingLanguage.PYTHON:
                result = self._execute_python(parameters, execution_dir, working_dir)
            elif parameters.language == ProgrammingLanguage.JAVASCRIPT:
                result = self._execute_javascript(parameters, execution_dir, working_dir)
            elif parameters.language == ProgrammingLanguage.JAVA:
                result = self._execute_java(parameters, execution_dir, working_dir)
            elif parameters.language == ProgrammingLanguage.CPP:
                result = self._execute_cpp(parameters, execution_dir, working_dir)
            elif parameters.language == ProgrammingLanguage.BASH:
                result = self._execute_bash(parameters, execution_dir, working_dir)
            else:
                # Default to generic execution
                result = self._execute_generic(parameters, execution_dir, working_dir)
        
        except Exception as e:
            # Handle unexpected errors
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(f"Error executing code: {e}")
            logger.error(traceback.format_exc())
            
            result = ExecutionResult(
                execution_id=execution_id,
                language=parameters.language,
                status=ExecutionStatus.SYSTEM_ERROR,
                stdout="",
                stderr="",
                execution_time=execution_time,
                error_message=str(e),
                additional_info={"traceback": traceback.format_exc()}
            )
        
        # Store result in history
        self.execution_history[execution_id] = result
        
        return result
    
    def _execute_python(self, parameters: ExecutionParameters, execution_dir: str, working_dir: str) -> ExecutionResult:
        """
        Execute Python code.
        
        Args:
            parameters: Execution parameters
            execution_dir: Directory for execution files
            working_dir: Working directory for execution
            
        Returns:
            Execution result
        """
        # Create Python file
        file_path = os.path.join(execution_dir, "code.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(parameters.code)
        
        # Create additional files
        for filename, content in parameters.additional_files.items():
            file_path = os.path.join(execution_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Prepare command
        cmd = ["python", "code.py"] + parameters.command_line_args
        
        # Execute command
        return self._run_command(
            cmd=cmd,
            execution_id=f"python_{int(time.time())}",
            language=ProgrammingLanguage.PYTHON,
            input_data=parameters.input_data,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
    
    def _execute_javascript(self, parameters: ExecutionParameters, execution_dir: str, working_dir: str) -> ExecutionResult:
        """
        Execute JavaScript code.
        
        Args:
            parameters: Execution parameters
            execution_dir: Directory for execution files
            working_dir: Working directory for execution
            
        Returns:
            Execution result
        """
        # Create JavaScript file
        file_path = os.path.join(execution_dir, "code.js")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(parameters.code)
        
        # Create additional files
        for filename, content in parameters.additional_files.items():
            file_path = os.path.join(execution_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Prepare command
        cmd = ["node", "code.js"] + parameters.command_line_args
        
        # Execute command
        return self._run_command(
            cmd=cmd,
            execution_id=f"javascript_{int(time.time())}",
            language=ProgrammingLanguage.JAVASCRIPT,
            input_data=parameters.input_data,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
    
    def _execute_java(self, parameters: ExecutionParameters, execution_dir: str, working_dir: str) -> ExecutionResult:
        """
        Execute Java code.
        
        Args:
            parameters: Execution parameters
            execution_dir: Directory for execution files
            working_dir: Working directory for execution
            
        Returns:
            Execution result
        """
        # Extract class name from code
        class_name = "Main"  # Default class name
        class_pattern = r"public\s+class\s+(\w+)"
        match = re.search(class_pattern, parameters.code)
        if match:
            class_name = match.group(1)
        
        # Create Java file
        file_path = os.path.join(execution_dir, f"{class_name}.java")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(parameters.code)
        
        # Create additional files
        for filename, content in parameters.additional_files.items():
            file_path = os.path.join(execution_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Compile Java code
        compile_cmd = ["javac", f"{class_name}.java"]
        compile_result = self._run_command(
            cmd=compile_cmd,
            execution_id=f"java_compile_{int(time.time())}",
            language=ProgrammingLanguage.JAVA,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
        
        # Check if compilation was successful
        if compile_result.status != ExecutionStatus.SUCCESS:
            return ExecutionResult(
                execution_id=f"java_{int(time.time())}",
                language=ProgrammingLanguage.JAVA,
                status=ExecutionStatus.COMPILATION_ERROR,
                stdout="",
                stderr=compile_result.stderr,
                execution_time=compile_result.execution_time,
                compilation_output=compile_result.stdout + "\n" + compile_result.stderr,
                error_message="Compilation failed"
            )
        
        # Run Java code
        run_cmd = ["java", class_name] + parameters.command_line_args
        run_result = self._run_command(
            cmd=run_cmd,
            execution_id=f"java_run_{int(time.time())}",
            language=ProgrammingLanguage.JAVA,
            input_data=parameters.input_data,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
        
        # Add compilation output to result
        run_result.compilation_output = compile_result.stdout + "\n" + compile_result.stderr
        
        return run_result
    
    def _execute_cpp(self, parameters: ExecutionParameters, execution_dir: str, working_dir: str) -> ExecutionResult:
        """
        Execute C++ code.
        
        Args:
            parameters: Execution parameters
            execution_dir: Directory for execution files
            working_dir: Working directory for execution
            
        Returns:
            Execution result
        """
        # Create C++ file
        file_path = os.path.join(execution_dir, "code.cpp")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(parameters.code)
        
        # Create additional files
        for filename, content in parameters.additional_files.items():
            file_path = os.path.join(execution_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Compile C++ code
        output_path = os.path.join(execution_dir, "code")
        compile_cmd = ["g++", "code.cpp", "-o", "code", "-std=c++17"]
        compile_result = self._run_command(
            cmd=compile_cmd,
            execution_id=f"cpp_compile_{int(time.time())}",
            language=ProgrammingLanguage.CPP,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
        
        # Check if compilation was successful
        if compile_result.status != ExecutionStatus.SUCCESS:
            return ExecutionResult(
                execution_id=f"cpp_{int(time.time())}",
                language=ProgrammingLanguage.CPP,
                status=ExecutionStatus.COMPILATION_ERROR,
                stdout="",
                stderr=compile_result.stderr,
                execution_time=compile_result.execution_time,
                compilation_output=compile_result.stdout + "\n" + compile_result.stderr,
                error_message="Compilation failed"
            )
        
        # Run C++ code
        run_cmd = ["./code"] + parameters.command_line_args
        run_result = self._run_command(
            cmd=run_cmd,
            execution_id=f"cpp_run_{int(time.time())}",
            language=ProgrammingLanguage.CPP,
            input_data=parameters.input_data,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
        
        # Add compilation output to result
        run_result.compilation_output = compile_result.stdout + "\n" + compile_result.stderr
        
        return run_result
    
    def _execute_bash(self, parameters: ExecutionParameters, execution_dir: str, working_dir: str) -> ExecutionResult:
        """
        Execute Bash code.
        
        Args:
            parameters: Execution parameters
            execution_dir: Directory for execution files
            working_dir: Working directory for execution
            
        Returns:
            Execution result
        """
        # Create Bash file
        file_path = os.path.join(execution_dir, "code.sh")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(parameters.code)
        
        # Make the script executable
        os.chmod(file_path, 0o755)
        
        # Create additional files
        for filename, content in parameters.additional_files.items():
            file_path = os.path.join(execution_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        # Prepare command
        cmd = ["bash", "code.sh"] + parameters.command_line_args
        
        # Execute command
        return self._run_command(
            cmd=cmd,
            execution_id=f"bash_{int(time.time())}",
            language=ProgrammingLanguage.BASH,
            input_data=parameters.input_data,
            timeout=parameters.timeout,
            working_dir=working_dir,
            env_vars=parameters.environment_variables
        )
    
    def _execute_generic(self, parameters: ExecutionParameters, execution_dir: str, working_dir: str) -> ExecutionResult:
        """
        Execute code in an unsupported language (placeholder).
        
        Args:
            parameters: Execution parameters
            execution_dir: Directory for execution files
            working_dir: Working directory for execution
            
        Returns:
            Execution result
        """
        # Create a file with the code
        file_extension = parameters.language.value
        file_path = os.path.join(execution_dir, f"code.{file_extension}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(parameters.code)
        
        # Return a placeholder result
        return ExecutionResult(
            execution_id=f"generic_{int(time.time())}",
            language=parameters.language,
            status=ExecutionStatus.SYSTEM_ERROR,
            stdout="",
            stderr=f"Execution of {parameters.language.value} is not supported yet.",
            execution_time=0.0,
            error_message=f"Execution of {parameters.language.value} is not supported yet."
        )
    
    def _run_command(self, cmd: List[str], execution_id: str, language: ProgrammingLanguage,
                    input_data: Optional[str] = None, timeout: float = 10.0,
                    working_dir: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """
        Run a command and capture its output.
        
        Args:
            cmd: Command to run
            execution_id: Execution ID
            language: Programming language
            input_data: Input data to provide to the command
            timeout: Timeout in seconds
            working_dir: Working directory for the command
            env_vars: Environment variables for the command
            
        Returns:
            Execution result
        """
        # Start execution timer
        start_time = time.time()
        
        # Prepare environment variables
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        try:
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env=env,
                text=True
            )
            
            # Provide input data if available
            if input_data:
                stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            else:
                stdout, stderr = process.communicate(timeout=timeout)
            
            # Get return code
            return_code = process.returncode
            
            # Determine execution status
            if return_code == 0:
                status = ExecutionStatus.SUCCESS
                error_message = None
            else:
                status = ExecutionStatus.RUNTIME_ERROR
                error_message = f"Process exited with code {return_code}"
        
        except subprocess.TimeoutExpired:
            # Handle timeout
            process.kill()
            stdout, stderr = process.communicate()
            return_code = -1
            status = ExecutionStatus.TIMEOUT
            error_message = f"Execution timed out after {timeout} seconds"
        
        except Exception as e:
            # Handle other errors
            if process:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
            else:
                stdout = ""
                stderr = str(e)
                return_code = -1
            
            status = ExecutionStatus.SYSTEM_ERROR
            error_message = str(e)
        
        # End execution timer
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Create execution result
        result = ExecutionResult(
            execution_id=execution_id,
            language=language,
            status=status,
            stdout=stdout,
            stderr=stderr,
            execution_time=execution_time,
            return_code=return_code,
            error_message=error_message
        )
        
        return result
    
    def get_result(self, execution_id: str) -> Optional[ExecutionResult]:
        """
        Get an execution result by ID.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Execution result or None if not found
        """
        return self.execution_history.get(execution_id)
    
    def analyze_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Analyze an execution result using the General Shape Equation.
        
        Args:
            result: Execution result
            
        Returns:
            Analysis results
        """
        # Prepare variables for equation evaluation
        variables = {
            "execution_status": result.status.value,
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage or 0.0,
            "memory_limit": 1000.0,  # Default memory limit
            "code_quality": 0.8  # Placeholder for code quality
        }
        
        # Evaluate equation
        equation_result = self.equation.evaluate(variables)
        
        # Prepare analysis results
        analysis = {
            "execution_id": result.execution_id,
            "language": result.language.value,
            "status": result.status.value,
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "equation_result": equation_result,
            "summary": self._generate_summary(result)
        }
        
        return analysis
    
    def _generate_summary(self, result: ExecutionResult) -> str:
        """
        Generate a summary of the execution result.
        
        Args:
            result: Execution result
            
        Returns:
            Summary string
        """
        if result.status == ExecutionStatus.SUCCESS:
            return f"Execution successful in {result.execution_time:.2f} seconds."
        elif result.status == ExecutionStatus.COMPILATION_ERROR:
            return f"Compilation error: {result.error_message or 'Unknown error'}"
        elif result.status == ExecutionStatus.RUNTIME_ERROR:
            return f"Runtime error: {result.error_message or 'Unknown error'}"
        elif result.status == ExecutionStatus.TIMEOUT:
            return f"Execution timed out after {result.execution_time:.2f} seconds."
        else:
            return f"Execution failed with status {result.status.value}: {result.error_message or 'Unknown error'}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the executor to a dictionary.
        
        Returns:
            Dictionary representation of the executor
        """
        return {
            "execution_history": {
                execution_id: {
                    "execution_id": result.execution_id,
                    "language": result.language.value,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "return_code": result.return_code,
                    "error_message": result.error_message,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr)
                }
                for execution_id, result in self.execution_history.items()
            }
        }
    
    def to_json(self) -> str:
        """
        Convert the executor to a JSON string.
        
        Returns:
            JSON string representation of the executor
        """
        return json.dumps(self.to_dict(), indent=2)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create executor
    executor = MultiLanguageExecutor()
    
    # Execute Python code
    python_params = ExecutionParameters(
        language=ProgrammingLanguage.PYTHON,
        code="""
print("Hello, world!")
for i in range(5):
    print(f"Number: {i}")
"""
    )
    
    python_result = executor.execute_code(python_params)
    
    print("Python Execution Result:")
    print(f"Status: {python_result.status}")
    print(f"Execution Time: {python_result.execution_time:.2f} seconds")
    print(f"Output:\n{python_result.stdout}")
    
    # Execute JavaScript code
    js_params = ExecutionParameters(
        language=ProgrammingLanguage.JAVASCRIPT,
        code="""
console.log("Hello from JavaScript!");
for (let i = 0; i < 5; i++) {
    console.log(`Number: ${i}`);
}
"""
    )
    
    js_result = executor.execute_code(js_params)
    
    print("\nJavaScript Execution Result:")
    print(f"Status: {js_result.status}")
    print(f"Execution Time: {js_result.execution_time:.2f} seconds")
    print(f"Output:\n{js_result.stdout}")
    
    # Analyze results
    python_analysis = executor.analyze_result(python_result)
    print("\nPython Analysis:")
    print(f"Summary: {python_analysis['summary']}")
    
    js_analysis = executor.analyze_result(js_result)
    print("\nJavaScript Analysis:")
    print(f"Summary: {js_analysis['summary']}")
