"""
Test-Driven Iterative Agent (Refactored)
==========================================
Clean class-based architecture with improved naming and organization.

Key Improvements:
1. Class-based design instead of procedural functions
2. Clear, descriptive naming conventions  
3. Separation of concerns across focused classes
4. Shorter methods (< 50 lines each)
5. Full type hints throughout
6. Better error handling and logging
"""

import os
import sys
import subprocess
import textwrap
import requests
import time
import re
import ast
import traceback
import logging
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# Configuration
# =============================================================================

class ModelType(Enum):
    """LLM models for different tasks."""
    REASONING = "deepseek-ai/DeepSeek-V3-0324"
    CODING = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" 
    FAST = "deepseek-ai/DeepSeek-V3-0324"


@dataclass
class AgentConfig:
    """Agent configuration."""
    run_id: str
    sandbox_proxy_url: str
    timeout: int
    max_iterations: int = 10
    max_alternatives: int = 10
    parallel_fixes: int = 3  # Number of parallel fix attempts
    parallel_alternatives: int = 3  # Number of parallel alternative solutions
    enable_parallelism: bool = True  # Enable/disable multithreading
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Load from environment variables."""
        return cls(
            run_id=os.getenv("RUN_ID", str(uuid4())),
            sandbox_proxy_url=os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy"),
            timeout=int(os.getenv("AGENT_TIMEOUT", "2000")),
            parallel_fixes=int(os.getenv("PARALLEL_FIXES", "3")),
            parallel_alternatives=int(os.getenv("PARALLEL_ALTERNATIVES", "3")),
            # enable_parallelism=os.getenv("ENABLE_PARALLELISM", "true").lower() == "true"
            enable_parallelism=True
        )


@dataclass
class TestResults:
    """Test execution results."""
    total: int
    passed: int
    failed: int
    errors: int
    passed_tests: List[str]
    failed_tests: List[str]
    error_details: List[Dict[str, str]]
    raw_output: str
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.total > 0


class ProblemType(Enum):
    """Problem type."""
    CREATE = "create"
    FIX = "fix"


# =============================================================================
# Logging
# =============================================================================

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


logger = setup_logger(__name__)


# =============================================================================
# LLMClient - Manages all LLM communication
# =============================================================================

class LLMClient:
    """Handles LLM API calls with retry logic."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = setup_logger(f"{__name__}.LLMClient")
    
    def query_model(
        self,
        messages: List[Dict[str, str]],
        model: ModelType = ModelType.CODING,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> str:
        """
        Query LLM with automatic retries.
        
        Renamed from: call_llm
        Improves clarity about what the method does.
        """
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.config.sandbox_proxy_url.rstrip('/')}/api/inference",
                    json={
                        "run_id": self.config.run_id,
                        "model": model.value,
                        "temperature": temperature,
                        "messages": messages
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict) and 'choices' in result:
                        return result['choices'][0]['message']['content']
                    elif isinstance(result, str):
                        return result.strip()
                    return str(result)
                else:
                    self.logger.warning(f"HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                self.logger.warning(f"Error: {e} (attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed after {max_retries} attempts")


# =============================================================================
# FileManager - Handles file operations
# =============================================================================

class FileManager:
    """Manages file I/O, parsing, and validation."""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.FileManager")
    
    @staticmethod
    def extract_code_from_markdown(text: str) -> str:
        """
        Extract Python code from markdown blocks.
        
        Renamed from: extract_python_code
        More descriptive name.
        """
        pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()
    
    def parse_multi_file_response(self, text: str) -> Dict[str, str]:
        """
        Parse LLM response containing multiple files.
        
        Renamed from: parse_file_blocks
        More specific about what it parses.
        
        Expected format:
            filename.py
            <code>
            
            another_file.py
            <code>
        """
        files = {}
        
        # Extract from code blocks first
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            text = '\n\n'.join(code_blocks)
        
        lines = text.split('\n')
        current_file = None
        current_content = []
        
        for line in lines:
            stripped = line.strip()
            # Detect filename
            if stripped.endswith('.py') and len(stripped) < 100 and '(' not in stripped:
                if current_file and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        files[current_file] = content
                current_file = stripped
                current_content = []
            elif current_file:
                current_content.append(line)
        
        # Save last file
        if current_file and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                files[current_file] = content
        
        # Fallback: treat as main.py if looks like Python code
        if not files and text.strip():
            if any(kw in text for kw in ['def ', 'class ', 'import ']):
                files['main.py'] = text.strip()
        
        return files
    
    def write_files_to_disk(self, files: Dict[str, str]) -> List[str]:
        """
        Write multiple files to disk.
        
        Renamed from: write_files
        More explicit about what it does.
        """
        created = []
        for filename, content in files.items():
            try:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                with open(filename, 'w') as f:
                    f.write(content)
                created.append(filename)
                self.logger.info(f"Created: {filename}")
            except Exception as e:
                self.logger.error(f"Failed to write {filename}: {e}")
        return created
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Check Python code for syntax errors.
        
        Renamed from: check_syntax
        More descriptive name.
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def scan_codebase_structure(self) -> str:
        """
        Scan all Python files to get code structure.
        
        Renamed from: get_code_skeleton
        Much clearer about purpose.
        """
        result = []
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                        result.append(f"{file}\n{{\n{content}\n}}\n")
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")
        return "\n".join(result)
    
    def find_python_files(self, exclude_tests: bool = False) -> Dict[str, str]:
        """
        Find and read all Python files in current directory.
        
        Args:
            exclude_tests: If True, skip files with 'test_' in name
            
        Returns:
            Dictionary mapping file paths to their content
        """
        files = {}
        for path in Path('.').rglob('*.py'):
            if exclude_tests and 'test_' in path.name:
                continue
            try:
                files[str(path)] = path.read_text()
            except Exception as e:
                self.logger.warning(f"Could not read {path}: {e}")
        return files
    
    def find_existing_test_files(self) -> Dict[str, str]:
        """
        Check if test files already exist in the work directory.
        
        Returns:
            Dictionary mapping test file paths to their content.
            Empty dict if no test files found.
        """
        test_files = {}
        
        # Look for common test file patterns
        test_patterns = ['test_*.py', 'tests.py', '*_test.py']
        
        for pattern in test_patterns:
            for path in Path('.').glob(pattern):
                try:
                    content = path.read_text()
                    test_files[str(path)] = content
                    self.logger.info(f"Found existing test file: {path}")
                except Exception as e:
                    self.logger.warning(f"Could not read test file {path}: {e}")
        
        return test_files
    
    def format_test_files_for_context(self, test_files: Dict[str, str]) -> str:
        """
        Format test files content for LLM context.
        
        Args:
            test_files: Dictionary mapping file paths to their content
            
        Returns:
            Formatted string with test files content
        """
        if not test_files:
            return "No test files available."
        
        formatted_parts = ["\n=== EXISTING TEST FILES ==="]
        for filename, content in test_files.items():
            formatted_parts.append(f"\n{filename}:\n```python\n{content}\n```")
        formatted_parts.append("\n=== END TEST FILES ===")
        
        return "\n".join(formatted_parts)


# =============================================================================
# TestRunner - Executes tests and parses results
# =============================================================================

class TestRunner:
    """Executes pytest and parses results."""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.TestRunner")
    
    def execute_test_suite(self, test_file: Optional[str] = None, timeout: int = 30) -> TestResults:
        """
        Run pytest and return structured results.
        
        Renamed from: run_tests
        More formal, descriptive name.
        """
        try:
            # Find test files
            test_files = [
                str(f) for f in Path('.').glob('*.py')
                if f.name.startswith('test_') or f.name == 'tests.py'
            ]
            
            if not test_files and not test_file:
                return self._create_error_result("No test files found")
            
            # Build pytest command
            if test_file:
                cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
            elif test_files:
                cmd = ["python", "-m", "pytest"] + test_files + ["-v", "--tb=short"]
            else:
                cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            output = result.stdout + "\n" + result.stderr
            return self._parse_pytest_output(output)
            
        except subprocess.TimeoutExpired:
            return self._create_error_result("Test execution timed out")
        except Exception as e:
            return self._create_error_result(f"Test execution error: {e}")
    
    def _parse_pytest_output(self, output: str) -> TestResults:
        """Parse pytest output into structured format."""
        # Extract counts
        passed = self._extract_count(output, r'(\d+) passed')
        failed = self._extract_count(output, r'(\d+) failed')
        errors = self._extract_count(output, r'(\d+) error')
        
        # Extract test names
        passed_tests = self._extract_test_names(output, 'PASSED')
        failed_tests = self._extract_test_names(output, 'FAILED')
        
        # Extract error details
        error_details = self._extract_failure_details(output)
        
        return TestResults(
            total=passed + failed + errors,
            passed=passed,
            failed=failed,
            errors=errors,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_details=error_details,
            raw_output=output
        )
    
    @staticmethod
    def _extract_count(output: str, pattern: str) -> int:
        """Extract count from pytest output."""
        match = re.search(pattern, output)
        return int(match.group(1)) if match else 0
    
    @staticmethod
    def _extract_test_names(output: str, status: str) -> List[str]:
        """Extract test names by status."""
        tests = []
        for line in output.split('\n'):
            if f' {status}' in line:
                test_name = line.split('::')[-1].split(f' {status}')[0].strip()
                tests.append(test_name)
        return tests
    
    @staticmethod
    def _extract_failure_details(output: str) -> List[Dict[str, str]]:
        """Extract detailed failure information."""
        details = []
        if '=== FAILURES ===' not in output:
            return details
        
        failure_section = output.split('=== FAILURES ===')[-1]
        current_test = None
        error_lines = []
        
        for line in failure_section.split('\n')[:100]:
            if line.startswith('_'):
                if current_test and error_lines:
                    details.append({
                        "test": current_test,
                        "error": '\n'.join(error_lines[:5])
                    })
                current_test = line.strip('_ ')
                error_lines = []
            elif line.strip() and current_test:
                error_lines.append(line)
        
        if current_test and error_lines:
            details.append({
                "test": current_test,
                "error": '\n'.join(error_lines[:5])
            })
        
        return details
    
    @staticmethod
    def _create_error_result(error_msg: str) -> TestResults:
        """Create TestResults for error conditions."""
        return TestResults(
            total=0, passed=0, failed=0, errors=1,
            passed_tests=[], failed_tests=[],
            error_details=[{"test": "error", "error": error_msg}],
            raw_output=error_msg
        )


# =============================================================================
# Prompt Templates (centralized)
# =============================================================================

class PromptTemplates:
    """Centralized prompt templates with clear naming."""
    
    @staticmethod
    def solution_generation_prompt(problem: str, test_examples: str = "") -> str:
        """Generate prompt for initial solution."""
        return f"""You are an expert Python developer. Use step-by-step reasoning.

Problem:
{problem}
{test_examples}

STEP 1 - ANALYZE: Understand requirements, constraints, edge cases
STEP 2 - DESIGN: Plan classes/functions, architecture, data structures
STEP 3 - IMPLEMENT: Write clean code handling all cases
STEP 4 - VERIFY: Check requirements, edge cases, readability

Generate complete solution as:
filename.py
```python
<code>
```"""
    
    @staticmethod
    def test_generation_system_prompt() -> str:
        """System prompt for test generation (multi-step reasoning)."""
        return """You are an expert Python unittest testcase developer. 
    Important points:-
    - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
    - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
    
    You must respond directly with the test cases in the following format. 
    =========TEST_CASES
    <<test cases>>
    Do not include anything else. For Example:
    =========TEST_CASES
    # These tests are auto-generated with test data from:
    # https://github.com/xxxx.json
    # File last updated on 2023-07-19
    import unittest
    from main_module import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()
"""
    
    @staticmethod
    def test_generation_user_prompt(problem: str, files: str, code_skeleton: str) -> str:
        """User prompt for test generation with problem details."""
        return f"""Problem Statement:
{problem}

Files To Test: {files}

Code skeleton: 
{code_skeleton}

Generate the complete and correct testcases in python files.

STRICT REQUIREMENT: You **MUST** output the **file name** along with file content.
example:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```"""
    
    @staticmethod
    def test_validation_system_prompt() -> str:
        """System prompt for test validation."""
        return """You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

Important:
1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
3. Minimize all testcases as you have context and generation limit

If no invalid testcases are detected and covered all critical edge cases:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.

example:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```"""
    
    @staticmethod
    def test_validation_user_prompt(problem: str, files: str, code_skeleton: str, generated_code: str) -> str:
        """User prompt for test validation with generated code."""
        return f"""Problem statement: {problem}

Files To Test: {files}

Code skeleton: 
{code_skeleton}

Generated Test Code:
{generated_code}

Analyze this code for invalid testcases. Return ONLY the final Python test code."""
    
    @staticmethod
    def failure_fix_prompt(problem: str, code: str, failures: str, test_context: str = "") -> str:
        """Generate prompt for fixing failures."""
        test_section = f"\n{test_context}\n" if test_context else ""
        return f"""Debug and fix failing tests systematically.

Problem: {problem}
Current Code:
{code}
{test_section}
Test Failures:
{failures}

STEP 1 - UNDERSTAND: What tests expect vs actual behavior
STEP 2 - IDENTIFY ROOT CAUSE: Underlying bug
STEP 3 - DESIGN FIX: Minimal change addressing failures
STEP 4 - IMPLEMENT: Apply fix maintaining quality

Output COMPLETE fixed code as:
filename.py
```python
<fixed code>
```"""
    @staticmethod
    def fallback_system_prompt() -> str:
        """System prompt for fallback test generation (simpler approach)."""
        return """You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```"""
    
    @staticmethod
    def fallback_user_prompt(problem: str, files: str, code_skeleton: str) -> str:
        """User prompt for fallback test generation."""
        return f"""Problem Statement:
{problem}

Python files to test:
{files}

Code skeleton: 
{code_skeleton}

Generate the ground truth and edge case coveraging testcases."""


# =============================================================================
# SolutionGenerator - Creates initial code
# =============================================================================

class SolutionGenerator:
    """Generates initial solutions."""
    
    def __init__(self, llm_client: LLMClient, file_manager: FileManager):
        self.llm = llm_client
        self.files = file_manager
        self.logger = setup_logger(f"{__name__}.SolutionGenerator")
    
    def generate_solution(self, problem: str) -> Dict[str, str]:
        """
        Generate initial solution code.
        
        Renamed from: generate_solution (kept same as it's already good)
        """
        test_examples = self._extract_test_examples()
        prompt = PromptTemplates.solution_generation_prompt(problem, test_examples)
        
        try:
            self.logger.info("Generating solution...")
            response = self.llm.query_model(
                [{"role": "user", "content": prompt}],
                model=ModelType.REASONING,
                temperature=0.0
            )
            
            code = self.files.extract_code_from_markdown(response)
            files = self.files.parse_multi_file_response(code or response)
            
            # Validate syntax
            for filename, content in files.items():
                valid, error = self.files.validate_python_syntax(content)
                status = "✓" if valid else f"✗ {error}"
                self.logger.info(f"{filename}: {status}")
            
            return files
        except Exception as e:
            self.logger.error(f"Solution generation failed: {e}")
            return {}
    
    @staticmethod
    def _extract_test_examples() -> str:
        """Extract example tests if tests.py exists."""
        if not Path('tests.py').exists():
            return ""
        try:
            content = Path('tests.py').read_text()
            lines = content.split('\n')
            examples = [l for l in lines[:50] if l.strip().startswith('def test_')][:3]
            if examples:
                return f"\n\nExample tests:\n```python\n{chr(10).join(examples)}...\n```"
        except:
            pass
        return ""


# =============================================================================
# TestSuiteGenerator - Creates tests
# =============================================================================

class TestSuiteGenerator:
    """Generates test suites."""
    
    def __init__(self, llm_client: LLMClient, file_manager: FileManager):
        self.llm = llm_client
        self.files = file_manager
        self.logger = setup_logger(f"{__name__}.TestSuiteGenerator")
    
    def generate_test_suite(
        self,
        problem: str,
        files_to_test: str,
        code_skeleton: str,
        max_attempts: int = 10
    ) -> str:
        """
        Generate comprehensive test suite.
        
        Renamed from: generate_test_files, generate_testcases_with_multi_step_reasoning
        Much clearer name.
        """
        retry = 0
        while retry < max_attempts:
            try:
                self.logger.info("Starting test cases generation")
                
                # # Try multi-step reasoning first
                # test_code = self._generate_tests(problem, files_to_test, code_skeleton)
                
                # if test_code:
                #     # Validate the generated tests
                #     validated = self._validate_tests(problem, files_to_test, code_skeleton, test_code)
                #     cleaned = self._clean_response(validated)
                    
                #     # Check format
                #     if cleaned and cleaned.split('\n')[0].endswith('.py'):
                #         self.logger.info("Generated testcases successfully using multi-step reasoning")
                #         return cleaned
                
                # # Fallback: use simpler single-step approach
                # self.logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                response = self.llm.query_model(
                    [
                        {"role": "system", "content": PromptTemplates.fallback_system_prompt()},
                        {"role": "user", "content": PromptTemplates.fallback_user_prompt(problem, files_to_test, code_skeleton)}
                    ],
                    model=ModelType.CODING
                )
                
                # Clean up the response
                testcases = self._clean_response(response)
                
                if testcases and testcases.split('\n')[0].endswith('.py'):
                    self.logger.info("Generated testcases successfully using fallback approach")
                    return testcases
                
                retry += 1
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error generating test cases: {str(e)}")
                retry += 1
                time.sleep(2)
        
        if retry >= max_attempts:
            self.logger.error("Failed to generate test cases")
        return ""
    
    def _generate_tests(self, problem: str, files: str, skeleton: str) -> str:
        """Generate initial tests with multi-step reasoning."""
        return self.llm.query_model(
            [
                {"role": "system", "content": PromptTemplates.test_generation_system_prompt()},
                {"role": "user", "content": PromptTemplates.test_generation_user_prompt(problem, files, skeleton)}
            ],
            model=ModelType.CODING
        )
    
    def _validate_tests(self, problem: str, files: str, skeleton: str, tests: str) -> str:
        """Validate and improve tests."""
        return self.llm.query_model(
            [
                {"role": "system", "content": PromptTemplates.test_validation_system_prompt()},
                {"role": "user", "content": PromptTemplates.test_validation_user_prompt(problem, files, skeleton, tests)}
            ],
            model=ModelType.CODING
        )
    
    @staticmethod
    def _clean_response(response: str) -> str:
        """Remove markdown formatting."""
        cleaned = response.strip()
        for prefix in ['```python', '```']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        return cleaned.strip()


# =============================================================================
# CodeFixer - Fixes test failures  
# =============================================================================

class CodeFixer:
    """Analyzes and fixes test failures."""
    
    def __init__(self, llm_client: LLMClient, file_manager: FileManager):
        self.llm = llm_client
        self.files = file_manager
        self.logger = setup_logger(f"{__name__}.CodeFixer")
    
    def generate_fixes_for_failures(
        self,
        problem: str,
        test_results: TestResults,
        current_files: Dict[str, str],
        test_context: str = ""
    ) -> Dict[str, str]:
        """
        Generate fixes for failing tests.
        
        Renamed from: fix_test_failures
        More descriptive name.
        
        Args:
            problem: Problem statement
            test_results: Test execution results
            current_files: Current solution files
            test_context: Optional test files content for LLM context
        """
        failures = self._summarize_failures(test_results)
        current_code = "\n\n".join(f"{name}:\n{content}" for name, content in current_files.items())
        
        prompt = PromptTemplates.failure_fix_prompt(problem, current_code, failures, test_context)
        
        try:
            self.logger.info("Generating fixes...")
            response = self.llm.query_model(
                [{"role": "user", "content": prompt}],
                model=ModelType.REASONING,
                temperature=0.0
            )
            
            code = self.files.extract_code_from_markdown(response)
            return self.files.parse_multi_file_response(code or response)
            
        except Exception as e:
            self.logger.error(f"Fix generation failed: {e}")
            return {}
    
    def generate_multiple_fixes_parallel(
        self,
        problem: str,
        test_results: TestResults,
        current_files: Dict[str, str],
        num_alternatives: int = 3,
        test_context: str = ""
    ) -> List[Tuple[Dict[str, str], float, str]]:
        """
        Generate multiple fix candidates in parallel with different approaches.
        
        Args:
            problem: Problem statement
            test_results: Test execution results
            current_files: Current solution files
            num_alternatives: Number of parallel fix attempts
            test_context: Optional test files content for LLM context
            
        Returns:
            List of (fixes_dict, temperature, approach_description) tuples
        """
        self.logger.info(f"Generating {num_alternatives} fix alternatives in parallel...")
        
        with ThreadPoolExecutor(max_workers=num_alternatives) as executor:
            # Submit multiple fix generation tasks with different temperatures
            futures = []
            for i in range(num_alternatives):
                temperature = 0.0 if i == 0 else 0.3 + (i * 0.2)
                approach = f"Approach {i+1} (temp={temperature})"
                
                future = executor.submit(
                    self._generate_single_fix,
                    problem,
                    test_results,
                    current_files,
                    temperature,
                    approach,
                    test_context
                )
                futures.append((future, temperature, approach))
            
            # Collect results as they complete
            fixes = []
            for future, temp, approach in futures:
                try:
                    fix_result = future.result(timeout=120)
                    if fix_result:
                        fixes.append((fix_result, temp, approach))
                        self.logger.info(f"✓ {approach} completed")
                except Exception as e:
                    self.logger.warning(f"✗ {approach} failed: {e}")
            
            return fixes
    
    def _generate_single_fix(
        self,
        problem: str,
        test_results: TestResults,
        current_files: Dict[str, str],
        temperature: float,
        approach: str,
        test_context: str = ""
    ) -> Dict[str, str]:
        """Generate a single fix attempt with specified temperature."""
        failures = self._summarize_failures(test_results)
        current_code = "\n\n".join(f"{name}:\n{content}" for name, content in current_files.items())
        
        prompt = PromptTemplates.failure_fix_prompt(problem, current_code, failures, test_context)
        
        try:
            response = self.llm.query_model(
                [{"role": "user", "content": prompt}],
                model=ModelType.REASONING,
                temperature=temperature
            )
            
            code = self.files.extract_code_from_markdown(response)
            return self.files.parse_multi_file_response(code or response)
            
        except Exception as e:
            self.logger.debug(f"Single fix generation failed ({approach}): {e}")
            return {}
    
    @staticmethod
    def _summarize_failures(results: TestResults) -> str:
        """Summarize test failures concisely."""
        if not results.failed_tests:
            return "No failures"
        
        summary = [f"Failed: {', '.join(results.failed_tests)}"]
        for detail in results.error_details[:3]:
            summary.append(f"\n{detail['test']}:\n{detail['error']}")
        
        return "\n".join(summary)


# =============================================================================
# TestDrivenAgent - Main orchestrator
# =============================================================================

class TestDrivenAgent:
    """
    Main agent orchestrating test-driven development.
    
    This replaces the procedural functions with a clean class-based design.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.file_manager = FileManager()
        self.test_runner = TestRunner()
        self.solution_generator = SolutionGenerator(self.llm_client, self.file_manager)
        self.test_generator = TestSuiteGenerator(self.llm_client, self.file_manager)
        self.code_fixer = CodeFixer(self.llm_client, self.file_manager)
        self.logger = setup_logger(f"{__name__}.TestDrivenAgent")
        self.test_files_content = ""  # Store test files content for LLM context
    
    def solve_problem(self, problem: str, repo_dir: str = "repo") -> str:
        """
        Main entry point to solve a problem.
        
        Renamed from: agent_main
        More intuitive name.
        """
        self.logger.info("="*80)
        self.logger.info("TEST-DRIVEN AGENT (REFACTORED)")
        self.logger.info("="*80)
        
        # Setup workspace
        self._setup_workspace(repo_dir)
        
        # Detect problem type
        problem_type = self._detect_problem_type(problem)
        self.logger.info(f"Problem type: {problem_type.value}")
        
        start_time = time.time()
        timeout = self.config.timeout - 120
        
        try:
            # Route to appropriate workflow
            if problem_type == ProblemType.CREATE:
                patch = self._create_workflow(problem, timeout, start_time)
            else:
                patch = self._fix_workflow(problem, timeout, start_time)
            
            # Clean up
            subprocess.run(["git", "reset", "--hard"], capture_output=True)
            return patch
            
        except Exception as e:
            self.logger.error(f"Agent failed: {e}")
            traceback.print_exc()
            try:
                patch = self._get_git_patch()
                subprocess.run(["git", "reset", "--hard"], capture_output=True)
                return patch
            except:
                return ""
    
    def _create_workflow(self, problem: str, timeout: int, start_time: float) -> str:
        """
        CREATE workflow: Generate solution and refine iteratively.
        
        Renamed from: create_mode
        More descriptive.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("CREATE WORKFLOW - Test-Driven Development")
        self.logger.info("="*80)
        
        code_skeleton = self.file_manager.scan_codebase_structure()
        # Generate solution
        self.logger.info("\n[STEP 1] Generating solution...")
        solution_files = self.solution_generator.generate_solution(problem)
        if not solution_files:
            self.logger.error("Failed to generate solution")
            return ""
        
        created = self.file_manager.write_files_to_disk(solution_files)
        self.logger.info(f"Created {len(created)} files")
        
        # Check for existing test files first
        self.logger.info("\n[STEP 2] Checking for test suite...")
        existing_test_files = self.file_manager.find_existing_test_files()
        
        if existing_test_files:
            # Test files already exist - read and store them
            self.logger.info(f"Found {len(existing_test_files)} existing test file(s)")
            for test_file in existing_test_files.keys():
                self.logger.info(f"  - {test_file}")
            
            # Store test content for LLM context
            self.test_files_content = self.file_manager.format_test_files_for_context(existing_test_files)
            self.logger.info("Using existing test files")
        else:
            # No test files exist - generate them
            self.logger.info("No existing test files found - generating test suite...")
            
            test_code = self.test_generator.generate_test_suite(
                problem,
                ", ".join(created),
                code_skeleton
            )
            
            if test_code:
                test_files = self._extract_and_write_test_files(test_code)
                self.logger.info(f"Created {len(test_files)} test files")
                
                # Re-read the generated test files to store their content
                generated_test_files = self.file_manager.find_existing_test_files()
                self.test_files_content = self.file_manager.format_test_files_for_context(generated_test_files)
        
        # Iterative refinement
        self.logger.info("\n[STEP 3] Iterative refinement...")
        return self._refine_solution_iteratively(
            problem, solution_files, timeout, start_time
        )
    
    def _fix_workflow(self, problem: str, timeout: int, start_time: float) -> str:
        """
        FIX workflow: Find and fix bugs iteratively.
        
        Renamed from: fix_mode
        More descriptive.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("FIX WORKFLOW - Iterative Debugging")
        self.logger.info("="*80)
        
        # Find relevant files
        relevant_files = self.file_manager.find_python_files(exclude_tests=True)
        self.logger.info(f"Found {len(relevant_files)} relevant files")
        
        # Iterative refinement
        return self._refine_solution_iteratively(
            problem, relevant_files, timeout, start_time
        )
    
    def _refine_solution_iteratively(
        self,
        problem: str,
        solution_files: Dict[str, str],
        timeout: int,
        start_time: float
    ) -> str:
        """
        Iteratively refine solution based on test results.
        
        Renamed from: post_workflow
        Much clearer name about what it does.
        """
        self.logger.info("Starting iterative refinement...")
        
        previous_failures = set()
        stuck_count = 0
        
        for iteration in range(self.config.max_iterations):
            if time.time() - start_time > timeout - 60:
                self.logger.info("Timeout approaching, stopping")
                break
            
            self.logger.info(f"\n--- Iteration {iteration + 1}/{self.config.max_iterations} ---")
            
            # Run tests
            results = self.test_runner.execute_test_suite()
            self.logger.info(f"Tests: {results.passed}/{results.total} passed")
            
            if results.all_passed:
                self.logger.info("✓ All tests passed!")
                break
            
            # Check if stuck
            current_failures = set(results.failed_tests)
            if current_failures == previous_failures:
                stuck_count += 1
                if stuck_count >= 2 and self.config.enable_parallelism:
                    self.logger.warning("⚠️ Stuck on same failures - exploring alternative architectures...")
                    
                    # PARALLEL: Explore alternative solutions
                    alternatives = self._explore_alternative_solutions_parallel(
                        problem, results, max_alternatives=self.config.parallel_alternatives
                    )
                    
                    if alternatives:
                        # Test all alternatives in parallel and pick the best
                        best_solution, best_results, best_desc = self._test_multiple_solutions_parallel(alternatives)
                        
                        if best_results.passed > results.passed:
                            self.logger.info(f"✓ Found better alternative: {best_desc}")
                            solution_files = best_solution
                            self.file_manager.write_files_to_disk(best_solution)
                            stuck_count = 0
                            previous_failures = set()
                            continue
                        else:
                            self.logger.warning("No better alternative found")
                    
                    break
            else:
                stuck_count = 0
            previous_failures = current_failures
            
            # Generate fixes (parallel or single)
            if self.config.enable_parallelism:
                self.logger.info("Generating fixes in parallel...")
                fix_candidates = self.code_fixer.generate_multiple_fixes_parallel(
                    problem, results, solution_files, 
                    num_alternatives=self.config.parallel_fixes,
                    test_context=self.test_files_content
                )
                best_fix = self._select_best_fix(fix_candidates)
            else:
                self.logger.info("Generating fix...")
                best_fix = self.code_fixer.generate_fixes_for_failures(
                    problem, results, solution_files,
                    test_context=self.test_files_content
                )
            
            if not best_fix:
                self.logger.warning("Could not generate fix")
                break
            
            solution_files.update(best_fix)
            self.file_manager.write_files_to_disk(best_fix)
        
        return self._get_git_patch()
    
    def _explore_alternative_solutions_parallel(
        self,
        problem: str,
        test_results: TestResults,
        max_alternatives: int = 3
    ) -> List[Tuple[Dict[str, str], str]]:
        """
        When stuck on same failures, try different architectural approaches in parallel.
        
        Returns:
            List of (solution_files, approach_hint) tuples
        """
        self.logger.info(f"Exploring {max_alternatives} alternative solutions in parallel...")
        
        approaches = [
            "Try a completely different algorithmic approach",
            "Use simpler data structures and more direct logic",
            "Refactor with better separation of concerns",
        ]
        
        with ThreadPoolExecutor(max_workers=max_alternatives) as executor:
            futures = {}
            for i, hint in enumerate(approaches[:max_alternatives]):
                future = executor.submit(
                    self._generate_solution_with_hint,
                    problem,
                    hint,
                    test_results
                )
                futures[future] = hint
            
            solutions = []
            for future in as_completed(futures):
                hint = futures[future]
                try:
                    solution = future.result(timeout=120)
                    if solution:
                        solutions.append((solution, hint))
                        self.logger.info(f"✓ Alternative '{hint[:40]}...' completed")
                except Exception as e:
                    self.logger.warning(f"✗ Alternative '{hint[:40]}...' failed: {e}")
            
            return solutions
    
    def _generate_solution_with_hint(
        self,
        problem: str,
        hint: str,
        test_results: TestResults
    ) -> Dict[str, str]:
        """Generate a solution with a specific architectural hint."""
        failures = self.code_fixer._summarize_failures(test_results)
        
        prompt = f"""You are an expert Python developer. The current solution is failing tests.

Problem:
{problem}

Current Failures:
{failures}

Architectural Hint: {hint}

STEP 1 - ANALYZE: Understand why current approach is failing
STEP 2 - REDESIGN: Apply the architectural hint to create a better approach
STEP 3 - IMPLEMENT: Write complete solution handling all cases
STEP 4 - VERIFY: Check requirements and edge cases

Generate complete solution as:
filename.py
```python
<code>
```"""
        
        try:
            response = self.llm_client.query_model(
                [{"role": "user", "content": prompt}],
                model=ModelType.REASONING,
                temperature=0.3
            )
            
            code = self.file_manager.extract_code_from_markdown(response)
            return self.file_manager.parse_multi_file_response(code or response)
            
        except Exception as e:
            self.logger.debug(f"Alternative solution generation failed: {e}")
            return {}
    
    def _test_multiple_solutions_parallel(
        self,
        solution_candidates: List[Tuple[Dict[str, str], str]]
    ) -> Tuple[Dict[str, str], TestResults, str]:
        """
        Test multiple solution candidates in parallel and return the best.
        
        Args:
            solution_candidates: List of (solution_files, description) tuples
            
        Returns:
            Tuple of (best_solution, test_results, description)
        """
        self.logger.info(f"Testing {len(solution_candidates)} solutions in parallel...")
        
        def test_solution(solution: Dict[str, str], desc: str, work_dir: str) -> Tuple[Dict[str, str], TestResults, str]:
            """Test a solution in an isolated directory."""
            os.makedirs(work_dir, exist_ok=True)
            
            # Write files
            for filename, content in solution.items():
                if not filename.startswith('test_'):  # Don't overwrite test files
                    filepath = os.path.join(work_dir, filename)
                    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                    with open(filepath, 'w') as f:
                        f.write(content)
            
            # Copy test files from current directory
            current_dir = os.getcwd()
            for test_file in Path(current_dir).glob('test_*.py'):
                shutil.copy(test_file, work_dir)
            
            # Run tests in that directory
            original_dir = os.getcwd()
            try:
                os.chdir(work_dir)
                results = self.test_runner.execute_test_suite()
                return solution, results, desc
            finally:
                os.chdir(original_dir)
        
        # Create temp directories and test in parallel
        with ThreadPoolExecutor(max_workers=len(solution_candidates)) as executor:
            futures = []
            temp_dirs = []
            
            for i, (solution, desc) in enumerate(solution_candidates):
                work_dir = tempfile.mkdtemp(prefix=f"test_solution_{i}_")
                temp_dirs.append(work_dir)
                future = executor.submit(test_solution, solution, desc, work_dir)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    solution, test_results, desc = future.result()
                    results.append((solution, test_results, desc))
                    self.logger.info(f"'{desc[:40]}...' -> {test_results.passed}/{test_results.total} passed")
                except Exception as e:
                    self.logger.warning(f"Testing failed: {e}")
            
            # Cleanup
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Return best solution (highest pass rate, then lowest failure rate)
        if results:
            best = max(results, key=lambda x: (x[1].passed, -x[1].failed))
            self.logger.info(f"✓ Best solution: '{best[2][:40]}...' with {best[1].passed}/{best[1].total} passed")
            return best
        
        # Fallback to empty result
        return {}, TestResults(0, 0, 0, 1, [], [], [], "No valid results"), "No solutions"
    
    def _select_best_fix(
        self,
        fix_candidates: List[Tuple[Dict[str, str], float, str]]
    ) -> Optional[Dict[str, str]]:
        """
        Select the best fix from multiple candidates.
        
        For now, returns the first valid fix (temp=0.0).
        Could be enhanced to test all and pick best.
        """
        if not fix_candidates:
            return None
        
        # Sort by temperature (prefer deterministic fixes first)
        sorted_fixes = sorted(fix_candidates, key=lambda x: x[1])
        
        for fixes, temp, approach in sorted_fixes:
            if fixes:
                self.logger.info(f"Selected: {approach}")
                return fixes
        
        return None
    
    def _setup_workspace(self, repo_dir: str):
        """Initialize git repository."""
        repo_path = os.path.abspath(repo_dir)
        if os.path.exists(repo_path):
            os.chdir(repo_path)
            self.logger.info(f"Working directory: {repo_path}")
        
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], capture_output=True)
            subprocess.run(["git", "config", "user.email", "agent@test.com"], capture_output=True)
            subprocess.run(["git", "config", "user.name", "Agent"], capture_output=True)
            subprocess.run(["git", "add", "-A"], capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], capture_output=True)
            self.logger.info("Initialized git")
    
    @staticmethod
    def _detect_problem_type(problem: str) -> ProblemType:
        """Detect if problem is CREATE or FIX."""
        py_files = list(Path('.').rglob('*.py'))
        has_code = len(py_files) > 0
        
        fix_keywords = ['fix', 'bug', 'error', 'broken', 'incorrect', 'modify']
        create_keywords = ['create', 'implement', 'write', 'build', 'generate']
        
        problem_lower = problem.lower()
        fix_score = sum(1 for kw in fix_keywords if kw in problem_lower)
        create_score = sum(1 for kw in create_keywords if kw in problem_lower)
        
        return ProblemType.FIX if (has_code and fix_score > create_score) else ProblemType.CREATE
    
    @staticmethod
    def _get_git_patch() -> str:
        """Get git diff patch."""
        try:
            subprocess.run(["git", "add", "*.py"], capture_output=True)
            result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True, timeout=10)
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to get patch: {e}")
            return ""
    
    def _extract_and_write_test_files(self, test_code: str) -> List[str]:
        """Extract test files from generated code and write them."""
        files = self.file_manager.parse_multi_file_response(test_code)
        return self.file_manager.write_files_to_disk(files)


# =============================================================================
# Main Entry Point
# =============================================================================

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo") -> str:
    """
    Main entry point (kept for backward compatibility).
    
    Args:
        input_dict: Dict with 'problem_statement' key
        repo_dir: Repository directory
        
    Returns:
        Git diff patch
    """
    config = AgentConfig.from_env()
    agent = TestDrivenAgent(config)
    
    problem = input_dict.get("problem_statement", "")
    if not problem:
        raise ValueError("No problem_statement provided")
    
    return agent.solve_problem(problem, repo_dir)


if __name__ == "__main__":
    # Example usage
    test_input = {"problem_statement": "Create a function that reverses a string"}
    result = agent_main(test_input)
    print(f"\nResult:\n{result}")
