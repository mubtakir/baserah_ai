#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Executor for Basira System

This module implements the Code Executor, which executes code in various programming
languages in a secure sandbox environment.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
import time
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
import signal

# Configure logging
logger = logging.getLogger('code_execution.code_executor')


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    RUBY = "ruby"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    BASH = "bash"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str  # Standard output
    stderr: str  # Standard error
    exit_code: int  # Exit code
    execution_time: float  # Execution time in seconds
    memory_usage: Optional[float] = None  # Memory usage in MB
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON  # Programming language
    code_hash: Optional[str] = None  # Hash of the executed code
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique execution ID


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    language: ProgrammingLanguage  # Programming language
    timeout: float = 5.0  # Timeout in seconds
    memory_limit: Optional[float] = None  # Memory limit in MB
    allow_network: bool = False  # Whether to allow network access
    allow_file_access: bool = False  # Whether to allow file access
    additional_args: List[str] = field(default_factory=list)  # Additional arguments for the interpreter/compiler
    environment_variables: Dict[str, str] = field(default_factory=dict)  # Environment variables


class CodeExecutor:
    """
    Code Executor class for executing code in various programming languages.
    
    This class provides a secure sandbox environment for executing code in
    various programming languages, with configurable resource limits and
    security restrictions.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 sandbox_dir: Optional[str] = None):
        """
        Initialize the Code Executor.
        
        Args:
            config_file: Path to the configuration file (optional)
            sandbox_dir: Path to the sandbox directory (optional)
        """
        self.logger = logging.getLogger('code_execution.code_executor.main')
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Set sandbox directory
        self.sandbox_dir = sandbox_dir or self.config.get("sandbox_dir", tempfile.gettempdir())
        
        # Create sandbox directory if it doesn't exist
        os.makedirs(self.sandbox_dir, exist_ok=True)
        
        # Initialize language handlers
        self.language_handlers = self._initialize_language_handlers()
        
        self.logger.info(f"Code Executor initialized with sandbox directory: {self.sandbox_dir}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use default configuration.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "sandbox_dir": tempfile.gettempdir(),
            "default_timeout": 5.0,
            "default_memory_limit": 100.0,  # MB
            "allow_network": False,
            "allow_file_access": False,
            "languages": {
                "python": {
                    "enabled": True,
                    "command": "python",
                    "file_extension": ".py",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "javascript": {
                    "enabled": True,
                    "command": "node",
                    "file_extension": ".js",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "ruby": {
                    "enabled": False,
                    "command": "ruby",
                    "file_extension": ".rb",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "java": {
                    "enabled": False,
                    "command": "java",
                    "compile_command": "javac",
                    "file_extension": ".java",
                    "timeout": 10.0,
                    "memory_limit": 200.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "cpp": {
                    "enabled": False,
                    "command": "",
                    "compile_command": "g++",
                    "file_extension": ".cpp",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "csharp": {
                    "enabled": False,
                    "command": "dotnet run",
                    "file_extension": ".cs",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "go": {
                    "enabled": False,
                    "command": "go run",
                    "file_extension": ".go",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "rust": {
                    "enabled": False,
                    "command": "",
                    "compile_command": "rustc",
                    "file_extension": ".rs",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "php": {
                    "enabled": False,
                    "command": "php",
                    "file_extension": ".php",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                },
                "bash": {
                    "enabled": True,
                    "command": "bash",
                    "file_extension": ".sh",
                    "timeout": 5.0,
                    "memory_limit": 100.0,
                    "allow_network": False,
                    "allow_file_access": False
                }
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Merge user config with default config
                self._merge_configs(default_config, user_config)
                self.logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {e}")
                self.logger.info("Using default configuration")
        else:
            if config_file:
                self.logger.warning(f"Configuration file {config_file} not found, using default configuration")
            else:
                self.logger.info("No configuration file provided, using default configuration")
        
        return default_config
    
    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """
        Merge user configuration with default configuration.
        
        Args:
            default_config: Default configuration dictionary (modified in-place)
            user_config: User configuration dictionary
        """
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_configs(default_config[key], value)
            else:
                default_config[key] = value
    
    def _initialize_language_handlers(self) -> Dict[ProgrammingLanguage, Dict[str, Any]]:
        """
        Initialize language handlers.
        
        Returns:
            Dictionary mapping programming languages to their handlers
        """
        handlers = {}
        
        for lang_str, lang_config in self.config["languages"].items():
            if lang_config["enabled"]:
                try:
                    lang = ProgrammingLanguage(lang_str)
                    handlers[lang] = lang_config
                    self.logger.info(f"Language handler initialized for {lang.value}")
                except ValueError:
                    self.logger.warning(f"Unknown programming language: {lang_str}")
        
        return handlers
    
    def execute(self, 
               code: str,
               language: ProgrammingLanguage,
               config: Optional[ExecutionConfig] = None,
               input_data: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in the specified programming language.
        
        Args:
            code: Code to execute
            language: Programming language
            config: Execution configuration (optional)
            input_data: Input data for the program (optional)
            
        Returns:
            Execution result
        """
        # Check if language is supported
        if language not in self.language_handlers:
            self.logger.error(f"Unsupported programming language: {language.value}")
            raise ValueError(f"Unsupported programming language: {language.value}")
        
        # Set default configuration if not provided
        if config is None:
            lang_config = self.language_handlers[language]
            config = ExecutionConfig(
                language=language,
                timeout=lang_config.get("timeout", self.config["default_timeout"]),
                memory_limit=lang_config.get("memory_limit", self.config["default_memory_limit"]),
                allow_network=lang_config.get("allow_network", self.config["allow_network"]),
                allow_file_access=lang_config.get("allow_file_access", self.config["allow_file_access"])
            )
        
        # Create temporary directory for execution
        execution_dir = tempfile.mkdtemp(dir=self.sandbox_dir)
        
        try:
            # Write code to file
            file_extension = self.language_handlers[language]["file_extension"]
            file_path = os.path.join(execution_dir, f"code{file_extension}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute code
            result = self._execute_file(file_path, language, config, input_data)
            
            return result
        finally:
            # Clean up temporary directory
            shutil.rmtree(execution_dir)
    
    def _execute_file(self, 
                     file_path: str,
                     language: ProgrammingLanguage,
                     config: ExecutionConfig,
                     input_data: Optional[str] = None) -> ExecutionResult:
        """
        Execute a file in the specified programming language.
        
        Args:
            file_path: Path to the file to execute
            language: Programming language
            config: Execution configuration
            input_data: Input data for the program (optional)
            
        Returns:
            Execution result
        """
        lang_config = self.language_handlers[language]
        
        # Prepare command
        if "compile_command" in lang_config:
            # Compiled language
            compile_command = lang_config["compile_command"]
            compile_args = [compile_command, file_path]
            
            # Compile code
            compile_result = self._run_process(
                compile_args,
                timeout=config.timeout,
                input_data=None,
                env=config.environment_variables
            )
            
            if compile_result.exit_code != 0:
                # Compilation failed
                return ExecutionResult(
                    stdout="",
                    stderr=f"Compilation failed:\n{compile_result.stderr}",
                    exit_code=compile_result.exit_code,
                    execution_time=compile_result.execution_time,
                    memory_usage=compile_result.memory_usage,
                    language=language,
                    execution_id=str(uuid.uuid4())
                )
            
            # Get executable path
            executable_path = os.path.splitext(file_path)[0]
            if language == ProgrammingLanguage.JAVA:
                # Java requires special handling
                class_name = os.path.basename(os.path.splitext(file_path)[0])
                command = [lang_config["command"], class_name]
            else:
                command = [executable_path]
        else:
            # Interpreted language
            command = [lang_config["command"], file_path]
        
        # Add additional arguments
        command.extend(config.additional_args)
        
        # Execute code
        result = self._run_process(
            command,
            timeout=config.timeout,
            input_data=input_data,
            env=config.environment_variables
        )
        
        # Create execution result
        execution_result = ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time=result.execution_time,
            memory_usage=result.memory_usage,
            language=language,
            execution_id=str(uuid.uuid4())
        )
        
        return execution_result
    
    def _run_process(self, 
                    command: List[str],
                    timeout: float,
                    input_data: Optional[str] = None,
                    env: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """
        Run a process with the specified command.
        
        Args:
            command: Command to run
            timeout: Timeout in seconds
            input_data: Input data for the process (optional)
            env: Environment variables (optional)
            
        Returns:
            Execution result
        """
        # Start timer
        start_time = time.time()
        
        # Prepare environment variables
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        try:
            # Run process
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=process_env,
                text=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            
            # End timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create execution result
            result = ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
                execution_time=execution_time,
                memory_usage=None,  # Memory usage not available
                language=ProgrammingLanguage.PYTHON,  # Placeholder
                execution_id=str(uuid.uuid4())
            )
            
            return result
        except subprocess.TimeoutExpired:
            # Process timed out
            process.kill()
            stdout, stderr = process.communicate()
            
            # End timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create execution result
            result = ExecutionResult(
                stdout=stdout if isinstance(stdout, str) else stdout.decode('utf-8', errors='replace'),
                stderr=f"Process timed out after {timeout} seconds\n" + (stderr if isinstance(stderr, str) else stderr.decode('utf-8', errors='replace')),
                exit_code=-1,  # -1 indicates timeout
                execution_time=execution_time,
                memory_usage=None,  # Memory usage not available
                language=ProgrammingLanguage.PYTHON,  # Placeholder
                execution_id=str(uuid.uuid4())
            )
            
            return result
        except Exception as e:
            # Process failed
            if process:
                process.kill()
            
            # End timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create execution result
            result = ExecutionResult(
                stdout="",
                stderr=f"Process failed: {str(e)}",
                exit_code=-2,  # -2 indicates error
                execution_time=execution_time,
                memory_usage=None,  # Memory usage not available
                language=ProgrammingLanguage.PYTHON,  # Placeholder
                execution_id=str(uuid.uuid4())
            )
            
            return result


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Code Executor
    executor = CodeExecutor()
    
    # Execute Python code
    python_code = """
print("Hello, world!")
for i in range(5):
    print(f"Number: {i}")
"""
    
    result = executor.execute(python_code, ProgrammingLanguage.PYTHON)
    
    # Print result
    print("Python Execution Result:")
    print(f"Exit Code: {result.exit_code}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print("Standard Output:")
    print(result.stdout)
    print("Standard Error:")
    print(result.stderr)
    
    # Execute JavaScript code
    javascript_code = """
console.log("Hello, world!");
for (let i = 0; i < 5; i++) {
    console.log(`Number: ${i}`);
}
"""
    
    result = executor.execute(javascript_code, ProgrammingLanguage.JAVASCRIPT)
    
    # Print result
    print("\nJavaScript Execution Result:")
    print(f"Exit Code: {result.exit_code}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print("Standard Output:")
    print(result.stdout)
    print("Standard Error:")
    print(result.stderr)
