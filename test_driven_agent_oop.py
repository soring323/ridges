"""
Test-Driven Agent - Object-Oriented Architecture

SOLID Principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Easy to extend with new strategies
- Liskov Substitution: Interfaces can be swapped
- Interface Segregation: Focused interfaces
- Dependency Inversion: Depend on abstractions
"""

import time
import os
import subprocess
import requests
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing
import logging
import sys
import textwrap

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ============================================================================
# Thread-Safety Infrastructure
# ============================================================================

# Module-level locks for shared state
_git_init_lock = Lock()  # Protect git initialization
_main_work_dir = None    # Store main working directory (set once at startup)

def set_main_work_dir(path: str):
    """Set main working directory once at agent startup (before threading)."""
    global _main_work_dir
    if _main_work_dir is None:
        _main_work_dir = os.path.abspath(path)
        logger.info(f"[INIT] Main work dir set to: {_main_work_dir}")

def get_main_work_dir() -> str:
    """Get the main working directory (thread-safe read)."""
    return _main_work_dir or os.getcwd()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ============================================================================
# Configuration (from test_driven_agent.py)
# ============================================================================

RUN_ID = os.getenv("RUN_ID", str(uuid4()))
SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))
ENABLE_PARALLEL = os.getenv("ENABLE_PARALLEL", "true").lower() == "true"  # Default: enabled

# Model selection
REASONING_MODEL = "deepseek-ai/DeepSeek-V3-0324"
CODING_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
FAST_MODEL = "deepseek-ai/DeepSeek-V3-0324"

MAX_ITERATIONS = 30


GENERATE_INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

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
```
"""
)

GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python unittest testcase developer familiar with canonical programming problem test specifications.

CRITICAL INSTRUCTIONS:
1. Many programming problems have canonical/standardized test data with precise input/output pairs
2. If you recognize this problem from your training data, recall the exact canonical test cases
3. Use the standard test data format with all edge cases and boundary conditions
4. Include a reference header citing the canonical test data source (as shown in example)
5. **MANDATORY: ALL imports must be from the 'main' module ONLY** (e.g., `from main import ...`)
6. **MANDATORY: Test file MUST be named 'tests.py' or 'test_something.py' (NEVER 'main.py')**

IMPORTANT: 
    - Search your knowledge for canonical test specifications for this problem
    - Use EXACT test data (inputs, outputs, edge cases) if you recall them
    - Follow professional test file conventions with proper documentation headers
    - You have generation limit of 2048 tokens. Stop generating when near the limit.
    - If you get syntax error and response was truncated, skip last couple of test cases to fit
    - **CRITICAL: Always import from 'main' module, never use other module names**
    
You must respond directly with the test cases in the following format:
=========TEST_CASES
<<test cases>>

Example format (include canonical data reference if applicable):
=========TEST_CASES
tests.py
# These tests are auto-generated with test data from:
# https://github.com/.../xxxxxxxxx-xxxx.json
# File last updated on YYYY-MM-DD
import unittest
from main import (
    main_func
)

class TestFuncA(unittest.TestCase):
    def test_main_func(self):
        self.assertEqual(main_func(), "expected_output")

if __name__ == "__main__":
    unittest.main()
"""
)


TESTCASES_CHECK_PROMPT = textwrap.dedent(
"""
You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

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
```
"""
)

# ============================================================================
# Resource Management & Parallel Execution
# ============================================================================

class ResourceManager:
    """Manages system resources and determines optimal thread count."""
    
    @staticmethod
    def get_optimal_workers() -> int:
        """Calculate optimal number of worker threads based on current CPU status."""
        cpu_count = multiprocessing.cpu_count()
        
        # Get load average (1-minute average)
        try:
            load_avg_1min = os.getloadavg()[0]
        except (AttributeError, OSError):
            # getloadavg() not available on Windows - use conservative default
            load_avg_1min = cpu_count * 0.5
        
        # Calculate available CPU capacity
        # If CPU is heavily loaded, reduce thread count
        cpu_load_ratio = load_avg_1min / cpu_count if cpu_count > 0 else 1.0
        
        # Determine thread count based on current load
        if cpu_load_ratio < 0.3:
            # Low load: use 75% of CPUs
            base_workers = int(cpu_count * 0.75)
        elif cpu_load_ratio < 0.6:
            # Medium load: use 50% of CPUs
            base_workers = int(cpu_count * 0.5)
        else:
            # High load: use 25% of CPUs
            base_workers = int(cpu_count * 0.25)
        
        # Apply bounds: minimum 2, maximum 8
        optimal = max(2, min(8, base_workers))
        
        print(f"[RESOURCE] CPU cores: {cpu_count}, "
              f"load avg: {load_avg_1min:.2f}, using {optimal} worker threads")
        return optimal

@dataclass
class SolutionCandidate:
    """A solution candidate with its test results."""
    solution_files: Dict[str, str]
    test_results: Optional[Any]  # TestResults, but defined later
    architecture_name: str
    generation_time: float
    
    @property
    def score(self) -> int:
        """Score based on tests passed."""
        return self.test_results.passed if self.test_results else 0
    
    @property
    def is_perfect(self) -> bool:
        """Check if solution passes all tests."""
        return self.test_results and self.test_results.success

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TestResults:
    """Test execution results."""
    passed: int
    failed: int
    total: int
    passed_tests: List[str]
    failed_tests: List[str]
    error_details: List[Dict[str, str]]
    
    @property
    def success(self) -> bool:
        """Success means: at least one test passed AND no tests failed."""
        return self.passed > 0 and self.failed == 0

@dataclass
class AlternativeArchitecture:
    """Alternative architecture proposal."""
    name: str
    files: Dict[str, str]
    score: int = 0
    
@dataclass
class RefinementConfig:
    """Configuration for refinement process."""
    max_iterations: int = 15  # Increased - give high-scoring solutions more attempts
    stuck_threshold: int = 5  # Increased - don't give up on almost-perfect solutions
    timeout: int = 1800

class ProblemMode(Enum):
    """Problem solving mode."""
    CREATE = "create"
    FIX = "fix"

# ============================================================================
# Core Interfaces
# ============================================================================

class ITestRunner(ABC):
    """Interface for test execution."""
    
    @abstractmethod
    def run_tests(self) -> Tuple[bool, str]:
        """Run tests and return (success, output)."""
        pass
    
    @abstractmethod
    def parse_results(self, output: str) -> TestResults:
        """Parse test output into structured results."""
        pass

class ICodeGenerator(ABC):
    """Interface for code generation."""
    
    @abstractmethod
    def generate_solution(self, problem: str, architecture_hint: Optional[str] = None, failure_hints: Optional[str] = None) -> Dict[str, str]:
        """Generate initial solution with optional architecture hint for diversity and failure hints from previous attempts."""
        pass
    
    @abstractmethod
    def generate_tests(self, problem: str, solution: Dict[str, str]) -> Dict[str, str]:
        """Generate test suite."""
        pass
    
    @abstractmethod
    def fix_failures(self, problem: str, test_output: str, files: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Generate fix for test failures."""
        pass

class IArchitectureGenerator(ABC):
    """Interface for alternative architecture generation."""
    
    @abstractmethod
    def generate_alternative(
        self, 
        problem: str, 
        current_files: Dict[str, str],
        test_results: TestResults,
        attempt: int
    ) -> Optional[AlternativeArchitecture]:
        """Generate alternative architecture."""
        pass

class IFileManager(ABC):
    """Interface for file operations."""
    
    @abstractmethod
    def write_files(self, files: Dict[str, str]) -> List[str]:
        """Write files to disk."""
        pass
    
    @abstractmethod
    def read_files(self, patterns: List[str]) -> Dict[str, str]:
        """Read files matching patterns."""
        pass

# ============================================================================
# Helper Functions (copied from test_driven_agent.py)
# ============================================================================

def truncate_text(text: str, max_chars: int = 50000) -> str:
    """Truncate text to avoid exceeding context limits."""
    if len(text) <= max_chars:
        return text
    
    # Keep first and last parts
    keep_each = max_chars // 2
    return (
        text[:keep_each] + 
        f"\n\n... [TRUNCATED {len(text) - max_chars} chars] ...\n\n" + 
        text[-keep_each:]
    )

def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4

def truncate_messages(messages: List[Dict], max_tokens: int = 200000) -> List[Dict]:
    """Truncate messages to fit within token limit."""
    truncated = []
    total_tokens = 0
    
    for msg in messages:
        content = msg.get("content", "")
        tokens = estimate_tokens(content)
        
        if total_tokens + tokens > max_tokens:
            # Truncate this message to fit
            remaining_tokens = max_tokens - total_tokens
            remaining_chars = remaining_tokens * 4
            
            if remaining_chars > 1000:  # Only include if meaningful
                truncated_content = truncate_text(content, remaining_chars)
                truncated.append({"role": msg["role"], "content": truncated_content})
            break
        
        truncated.append(msg)
        total_tokens += tokens
    
    return truncated

def call_llm(messages: List[Dict], model: str = CODING_MODEL, temperature: float = 0.0, max_retries: int = 5) -> str:
    """Call LLM with enhanced retry logic and token limit protection."""
    # Truncate messages to avoid context overflow
    # Use 120k tokens max to be safe for smaller models (DeepSeek: 163k context)
    messages = truncate_messages(messages, max_tokens=120000)
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{SANDBOX_PROXY_URL.rstrip('/')}/api/inference",
                json={
                    "run_id": RUN_ID,
                    "model": model,
                    "temperature": temperature,
                    "messages": messages
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'choices' in result:
                    return result['choices'][0]['message']['content']
                elif isinstance(result, str):
                    return result.strip()
                else:
                    return str(result)
            elif response.status_code == 429:
                # Rate limit - wait longer before retry
                last_error = "Rate limit (429)"
                wait_time = min(60, 10 * (2 ** attempt))  # 10s, 20s, 40s, 60s...
                print(f"[RATE LIMIT] 429 Too Many Requests - waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue
            else:
                last_error = f"HTTP {response.status_code}"
                print(f"[NETWORK] HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
                
        except requests.exceptions.ConnectionError:
            last_error = "Connection refused"
            print(f"[NETWORK] Connection refused (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = min(30, 5 * (2 ** attempt))
                time.sleep(wait_time)
                continue
                
        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            print(f"[NETWORK] Timeout after 180s (attempt {attempt + 1}/{max_retries})")
            
        except Exception as e:
            last_error = str(e)
            print(f"[NETWORK] Error: {e} (attempt {attempt + 1}/{max_retries})")
        
        if attempt < max_retries - 1:
            backoff = 2 ** attempt
            time.sleep(backoff)
    
    error_msg = f"Failed to get LLM response after {max_retries} attempts. Last error: {last_error}"
    print(f"[NETWORK] {error_msg}")
    raise RuntimeError(error_msg)

def parse_file_blocks(text: str) -> Dict[str, str]:
    """Parse text containing multiple files."""
    files = {}
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        text = '\n\n'.join(code_blocks)
    
    lines = text.split('\n')
    current_file = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.endswith('.py') and len(stripped) < 100 and '(' not in stripped:
            if current_file and current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    files[current_file] = content
            current_file = stripped
            current_content = []
        elif current_file:
            current_content.append(line)
    
    if current_file and current_content:
        content = '\n'.join(current_content).strip()
        if content:
            files[current_file] = content
    
    if not files and text.strip():
        if 'def ' in text or 'class ' in text or 'import ' in text:
            files['main.py'] = text.strip()
    
    return files

def write_files_helper(files: Dict[str, str]) -> List[str]:
    """Write files to disk."""
    created = []
    for filename, content in files.items():
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                f.write(content)
            created.append(filename)
        except Exception as e:
            print(f"[FILES] Error writing {filename}: {e}")
    return created

def run_tests_helper(test_file: Optional[str] = None, timeout: int = 30, work_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Run tests and return (success, output).
    
    Args:
        test_file: Specific test file to run
        timeout: Test timeout in seconds
        work_dir: Working directory for tests (explicit, not cwd!)
    """
    try:
        if work_dir is None:
            work_dir = os.getcwd()
        
        test_files_found = []
        for f in Path(work_dir).glob('*.py'):
            if f.name.startswith('test_') or f.name == 'tests.py':
                test_files_found.append(str(f.name))  # Use relative name
        
        if not test_files_found and not test_file:
            return False, "[TEST ERROR] No test files found"
        
        if test_file:
            cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
        else:
            if test_files_found:
                cmd = ["python", "-m", "pytest"] + test_files_found + ["-v", "--tb=short"]
            else:
                cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout, 
            cwd=work_dir  # EXPLICIT CWD!
        )
        output = result.stdout + "\n" + result.stderr
        success = result.returncode == 0
        
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"[TEST ERROR] Tests timed out after {timeout}s"
    except Exception as e:
        return False, f"[TEST ERROR] {str(e)}"

def parse_test_results_helper(output: str) -> Dict[str, Any]:
    """Parse pytest output into structured results."""
    results = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'total': 0,
        'passed_tests': [],
        'failed_tests': [],
        'error_details': []
    }
    
    # Parse test counts using regex (more reliable)
    # Look for patterns like "12 failed, 2 passed" or "2 passed" or "12 failed"
    passed_match = re.search(r'(\d+) passed', output)
    failed_match = re.search(r'(\d+) failed', output)
    error_match = re.search(r'(\d+) error', output)
    
    results['passed'] = int(passed_match.group(1)) if passed_match else 0
    results['failed'] = int(failed_match.group(1)) if failed_match else 0
    results['errors'] = int(error_match.group(1)) if error_match else 0
    results['total'] = results['passed'] + results['failed'] + results['errors']
    
    # Parse individual test results
    for line in output.split('\n'):
        if ' PASSED' in line and '::' in line:
            # Extract test name from format: tests.py::ReactTest::test_name PASSED
            test_name = line.split('::')[-1].split(' PASSED')[0].split(' [')[0].strip()
            if test_name and test_name not in results['passed_tests']:
                results['passed_tests'].append(test_name)
        elif ' FAILED' in line and '::' in line:
            # Extract test name from format: tests.py::ReactTest::test_name FAILED
            test_name = line.split('::')[-1].split(' FAILED')[0].split(' [')[0].split(' - ')[0].strip()
            if test_name and test_name not in results['failed_tests']:
                results['failed_tests'].append(test_name)
    
    # Extract detailed error messages from FAILURES section
    if '=== FAILURES ===' in output or '=== ERRORS ===' in output:
        failure_section = output.split('=== FAILURES ===')[-1] if '=== FAILURES ===' in output else ""
        if failure_section:
            # Extract error details for each test
            current_test = None
            error_lines = []
            for line in failure_section.split('\n')[:300]:  # Increased from 200 to 300
                if line.startswith('_'):  # Test separator like "_ test_name _"
                    # Save previous test's error
                    if current_test and error_lines:
                        results['error_details'].append({
                            'test': current_test,
                            'error': '\n'.join(error_lines[:30])  # Increased from 10 to 30 lines
                        })
                    # Start new test
                    current_test = line.strip('_ ')
                    error_lines = []
                elif line.strip() and current_test:
                    error_lines.append(line)
            
            # Add last test's error
            if current_test and error_lines:
                results['error_details'].append({
                    'test': current_test,
                    'error': '\n'.join(error_lines[:30])  # Increased from 10 to 30 lines
                })
    
    # Debug output
    if results['total'] > 0:
        print(f"[PARSE] Found {results['passed']} passed, {results['failed']} failed, {results['total']} total tests")
    
    return results

def ensure_git_initialized():
    """Initialize git repository if not already initialized (THREAD-SAFE)."""
    with _git_init_lock:  # Serialize all git initialization
        work_dir = get_main_work_dir()  # Use stored directory, never getcwd()!
        
        logger.info(f"[GIT] Checking git initialization in {work_dir}")
        
        git_dir = os.path.join(work_dir, ".git")
        
        # Check if already initialized
        if os.path.exists(git_dir):
            logger.info("[GIT] Repository already initialized")
            # Ensure safe directory (idempotent)
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", work_dir],
                check=False  # Don't fail if already exists
            )
            return
        
        # Initialize new repository
        logger.info("[GIT] Initializing new repository")
        
        try:
            # Use explicit cwd parameter instead of chdir
            subprocess.run(
                ["git", "init"],
                cwd=work_dir,
                check=True,
                capture_output=True
            )
            
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", work_dir],
                check=True
            )
            
            # Set local git config
            subprocess.run(
                ["git", "config", "user.email", "agent@sandbox.local"],
                cwd=work_dir,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "sandbox_agent"],
                cwd=work_dir,
                check=True
            )
            
            # Add and commit all files
            subprocess.run(
                ["git", "add", "."],
                cwd=work_dir,
                check=True
            )
            
            result = subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=work_dir,
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("[GIT] Initial commit created")
            else:
                logger.info(f"[GIT] Commit result: {result.stderr.strip()}")
            
            logger.info("[GIT] Initialization completed")
            
        except Exception as e:
            logger.error(f"[GIT] ERROR: Could not initialize git repository: {e}")


# ============================================================================
# Concrete Implementations
# ============================================================================

class PytestRunner(ITestRunner):
    """Concrete implementation of test runner using pytest."""
    
    def run_tests(self) -> Tuple[bool, str]:
        """Run tests using helper function."""
        return run_tests_helper()
    
    def parse_results(self, output: str) -> TestResults:
        """Parse test results using helper function."""
        results_dict = parse_test_results_helper(output)
        return TestResults(
            passed=results_dict['passed'],
            failed=results_dict['failed'],
            total=results_dict['total'],
            passed_tests=results_dict['passed_tests'],
            failed_tests=results_dict['failed_tests'],
            error_details=results_dict.get('error_details', [])
        )

class LLMCodeGenerator(ICodeGenerator):
    """Concrete implementation using LLM API."""
    
    def generate_solution(self, problem: str, architecture_hint: Optional[str] = None, failure_hints: Optional[str] = None) -> Dict[str, str]:
        """Generate solution using LLM with optional architecture hint for diversity and failure hints from previous attempts."""
        # Truncate problem statement to avoid context overflow
        problem = truncate_text(problem, max_chars=5000)
        
        # Check if tests.py exists to show examples
        test_examples = ""
        if Path('tests.py').exists():
            with open('tests.py', 'r') as f:
                test_content = f.read()
                # Limit test content to 5k chars
                test_content = truncate_text(test_content, max_chars=5000)
                test_lines = test_content.split('\n')
                example_lines = []
                in_test = False
                test_count = 0
                for line in test_lines:
                    if line.strip().startswith('def test_') and test_count < 3:
                        in_test = True
                        test_count += 1
                    if in_test:
                        example_lines.append(line)
                        if line.strip() and not line.strip().startswith('#') and test_count >= 3:
                            if 'self.assert' in line or 'self.assertEqual' in line:
                                example_lines.append('        ...\n')
                                break
                if example_lines:
                    test_examples = f"\n\nExample tests you need to pass:\n```python\n{''.join(example_lines[:50])}\n```"
        
        # Add architecture hint if provided
        architecture_guidance = ""
        if architecture_hint:
            architecture_guidance = f"\n\n🎯 REQUIRED ARCHITECTURE APPROACH:\n{architecture_hint}\n\nYou MUST follow this architectural approach in your solution.\n"
        
        # Add failure hints if provided (critical for learning from previous attempts)
        failure_guidance = ""
        if failure_hints:
            failure_guidance = f"\n{failure_hints}\n"
        
        prompt = f"""You are an expert Python developer. Use step-by-step reasoning to solve this problem.

                    Problem Statement:
                    {problem}
                    {test_examples}
                    {architecture_guidance}
                    {failure_guidance}
                    
                    STEP 1 - ANALYZE THE PROBLEM:
                    - What is the core problem asking for?
                    - What are the key requirements and constraints?
                    - What edge cases need to be handled?
                    - What data structures or algorithms are most appropriate?
                    
                    STEP 2 - DESIGN THE SOLUTION:
                    - What classes/functions are needed?
                    - How should they interact?
                    - What is the overall architecture?{' (MUST align with the required approach above)' if architecture_hint else ''}
                    - What are potential pitfalls to avoid?
                    
                    STEP 3 - IMPLEMENT:
                    - Write clean, well-structured code
                    - Handle all edge cases identified in Step 1
                    - Use only standard library (no external dependencies)
                    - Include proper error handling
                    - Make sure the implementation matches the design
                    
                    STEP 4 - VERIFY:
                    - Does the solution address all requirements?
                    - Are edge cases handled?
                    - Is the code readable and maintainable?
                    
                    Now generate the complete solution in this format:
                    filename.py
                    ```python
                    <complete code>
                    ```
                    
                    Generate your solution:"""
        
        try:
            response = call_llm(
                [{"role": "user", "content": prompt}],
                model=REASONING_MODEL,
                temperature=0.0
            )
            
            files = parse_file_blocks(response)
            if not files:
                return {}
            
            return files
        except Exception as e:
            print(f"[ERROR] Solution generation failed: {e}")
            return {}
    
    def generate_tests(self, problem: str, solution: Dict[str, str]) -> Dict[str, str]:
        """Generate tests using LLM with improved canonical test recall prompt."""
        logger.info("="*80)
        logger.info("TEST GENERATION - Starting process")
        logger.info("="*80)
        
        # If no solution provided, generate tests based on problem statement alone
        if solution:
            solution_summary = "\n\n".join([f"{name}:\n{content[:500]}..." for name, content in solution.items()])
            solution_files_list = list(solution.keys())
            main_file = solution_files_list[0] if solution_files_list else "main.py"
            main_module = main_file.replace('.py', '') if main_file.endswith('.py') else main_file
        else:
            solution_summary = "(No solution provided - generate tests based on problem statement)"
            solution_files_list = []
            main_module = "main"  # Default module name
        
        logger.info(f"[TEST_GEN] Target module: {main_module}")
        logger.info(f"[TEST_GEN] Solution files: {solution_files_list if solution_files_list else 'None (generating from problem statement)'}")
        logger.info(f"[TEST_GEN] Problem statement (first 300 chars): {problem[:300]}...")
        
        # Step 1: Generate initial tests using improved prompt
        if solution:
            context_note = f"""Solution Files:
{solution_summary}

Generate comprehensive test cases for this solution."""
        else:
            context_note = """No solution provided yet. Generate canonical test cases based ONLY on the problem statement.
You MUST determine the correct module name and function signatures from the problem description."""
        
        user_message = f"""Problem Statement:
{problem}

{context_note}

Follow the format and instructions provided.

CRITICAL REQUIREMENTS:
- **MANDATORY: Import from 'main' module ONLY** (e.g., `from main import InputCell, ComputeCell`)
- **MANDATORY: Test file MUST be named 'tests.py' or 'test_*.py' (NOT main.py)**
- Use standard Python unittest or pytest
- The solution will always be in main.py, so tests must import from main

Example file structure:
```
tests.py  <- YOUR TEST FILE (REQUIRED NAME)
import unittest
from main import YourClass, your_function

class TestYourClass(unittest.TestCase):
    def test_example(self):
        self.assertEqual(your_function(), expected_value)
```

DO NOT name your test file 'main.py' - that is for the solution code!
"""
        
        try:
            # Step 1: Generate tests with canonical test recall prompt
            logger.info("[TEST_GEN] Step 1: Generating initial tests with canonical recall prompt...")
            logger.info(f"[TEST_GEN] Using model: {CODING_MODEL}")
            logger.info("[TEST_GEN] Temperature: 0.0 (deterministic)")
            logger.info("[TEST_GEN] System prompt: GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT")
            
            print("[TEST_GEN] Step 1: Generating initial tests...")
            response = call_llm(
                [
                    {"role": "system", "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                model=CODING_MODEL,
                temperature=0.0  # Deterministic for consistent canonical test recall
            )
            
            logger.info(f"[TEST_GEN] Step 1 - Received response ({len(response)} chars)")
            logger.info("="*80)
            logger.info("[TEST_GEN] FULL INITIAL RESPONSE FROM LLM:")
            logger.info("="*80)
            logger.info(response)
            logger.info("="*80)
            
            # Check if response contains canonical data reference
            if 'github.com' in response.lower() or 'canonical' in response.lower():
                logger.info("[TEST_GEN] ✓ Response contains canonical/GitHub reference - likely recalled standard tests!")
            else:
                logger.warning("[TEST_GEN] ⚠ Response does NOT contain canonical reference - may be generated from scratch")
            
            # Step 2: Validate and refine tests
            logger.info("[TEST_GEN] Step 2: Validating and refining tests...")
            print("[TEST_GEN] Step 2: Validating and refining tests...")
            validation_prompt = f"""You are an expert unittest testcase reviewer. Analyze the generated tests for validity.

                Problem Statement:
                {problem}

                Solution Files:
                {solution_summary}

                Generated Tests:
                {response}

                Check for:
                1. Incorrect/invalid input/output pairs based on problem statement - fix or remove them
                2. Missing critical edge cases - add them
                3. Minimize testcases (context limit)
                4. If a file is not testcase file, remove it.

                If tests are valid and complete:
                - Return the original tests unchanged

                STRICT REQUIREMENT: Return ONLY the final Python test code with file names.

                Example format (using actual solution file "{main_module}"):
                tests.py
                import pytest
                from {main_module} import function_name

                def test_basic():
                    assert function_name() == expected
                """
            
            validated_response = call_llm(
                [{"role": "user", "content": validation_prompt}],
                model=CODING_MODEL,
                temperature=0.0  # Deterministic for validation
            )
            
            logger.info(f"[TEST_GEN] Step 2 - Validation complete ({len(validated_response)} chars)")
            
            # Step 3: Parse test files
            logger.info("[TEST_GEN] Step 3: Parsing validated tests...")
            print("[TEST_GEN] Step 3: Parsing validated tests...")
            files = parse_file_blocks(validated_response)
            
            if not files:
                logger.warning("[TEST_GEN] Validation parsing failed, trying original response...")
                print("[TEST_GEN] Validation failed, using original tests...")
                files = parse_file_blocks(response)
            else:
                logger.info("[TEST_GEN] ✓ Tests validated and refined")
                print("[TEST_GEN] ✓ Tests validated and refined")
            
            # Log final results
            if files:
                logger.info(f"[TEST_GEN] ✓ Successfully generated {len(files)} test file(s): {list(files.keys())}")
                for filename, content in files.items():
                    num_tests = content.count('def test_')
                    logger.info(f"[TEST_GEN]   - {filename}: {num_tests} test functions, {len(content)} chars")
                    logger.info("="*80)
                    logger.info(f"[TEST_GEN] FULL CONTENT OF {filename}:")
                    logger.info("="*80)
                    logger.info(content)
                    logger.info("="*80)
            else:
                logger.error("[TEST_GEN] ✗ Failed to parse any test files from response")
            
            logger.info("="*80)
            return files if files else {}
            
        except Exception as e:
            logger.error(f"[TEST_GEN] ✗ Test generation failed with exception: {e}", exc_info=True)
            print(f"[ERROR] Test generation failed: {e}")
            return {}
    
    def _extract_failure_summary(self, test_output: str, max_lines: int = 500) -> str:
        """Extract detailed failure information from test output.
        
        This extracts the FULL FAILURES section with all error messages,
        stack traces, and assertions. This is CRITICAL for the LLM to
        understand what's wrong and fix it.
        """
        # Extract the FAILURES section which has the most detail
        if '=== FAILURES ===' in test_output:
            failures_start = test_output.index('=== FAILURES ===')
            failures_section = test_output[failures_start:]
            
            # Remove the short summary at the end (less useful, redundant)
            if '=== short test summary' in failures_section:
                summary_start = failures_section.index('=== short test summary')
                failures_section = failures_section[:summary_start]
            
            # Include as much context as possible (increased from 300 to 500 lines)
            # Each test failure typically needs 10-30 lines to show full context
            failure_lines = failures_section.split('\n')[:max_lines]
            result = '\n'.join(failure_lines)
            
            # If we hit the limit, add a note
            if len(failures_section.split('\n')) > max_lines:
                result += f"\n\n[NOTE: Output truncated at {max_lines} lines. Full output available if needed.]"
            
            return result
        
        # Fallback: return last part of output (increased from 2000 to 5000 chars)
        return test_output[-5000:] if len(test_output) > 5000 else test_output
    
    def fix_failures(self, problem: str, test_output: str, files: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Fix failures using LLM with detailed error context."""
        # Extract detailed failure information from FAILURES section
        # Limit to 200 lines to avoid context overflow
        failure_summary = self._extract_failure_summary(test_output, max_lines=200)
        
        # Detect if this is a large problem (SWE-Bench style)
        total_file_size = sum(len(content) for content in files.values())
        is_large_problem = total_file_size > 50000 or len(files) > 5
        
        # Adaptive truncation based on problem size
        if is_large_problem:
            # Aggressive truncation for large codebases
            max_file_chars = 3000  # Much smaller per file
            max_files = 5  # Only include most relevant files
            max_problem_chars = 2000
            print(f"[TRUNCATE] Large problem detected ({len(files)} files, {total_file_size} chars)")
            print(f"[TRUNCATE] Applying aggressive truncation: {max_files} files, {max_file_chars} chars each")
        else:
            # Normal truncation
            max_file_chars = 10000
            max_files = 20
            max_problem_chars = 5000
        
        # Truncate files to avoid context overflow
        truncated_files = {}
        for i, (name, content) in enumerate(files.items()):
            if i >= max_files:
                print(f"[TRUNCATE] Skipping {len(files) - max_files} additional files")
                break
            truncated_files[name] = truncate_text(content, max_chars=max_file_chars)
        
        files_summary = "\n\n".join([f"{name}:\n{content}" for name, content in truncated_files.items()])
        
        # Truncate problem statement too
        problem = truncate_text(problem, max_chars=max_problem_chars)
        
        prompt = f"""You are an expert debugger. Fix the failing tests by analyzing the detailed error messages.

                    Problem Statement:
                    {problem}

                    Current Code:
                    {files_summary}

                    Detailed Test Failures (with stack traces and assertions):
                    {failure_summary}

                    Instructions:
                    1. Read the FULL error messages and stack traces above
                    2. Identify the EXACT cause of each failure (wrong value, missing method, logic error, etc.)
                    3. Fix the code to address the root cause
                    4. Ensure all edge cases are handled
                    5. Don't break existing passing tests

                    CRITICAL: The error messages above show you EXACTLY what's wrong. Use them!

                    Output the fixed code in this format:
                    filename.py
                    ```python
                    <complete fixed code>
                    ```

                    Generate the fixed code:"""
        
        try:
            response = call_llm(
                [{"role": "user", "content": prompt}],
                model=CODING_MODEL,
                temperature=0.0
            )
            
            fixed_files = parse_file_blocks(response)
            return fixed_files if fixed_files else None
        except Exception as e:
            print(f"[ERROR] Fix generation failed: {e}")
            return None

class LLMArchitectureGenerator(IArchitectureGenerator):
    """Concrete implementation for generating alternative architectures."""
    
    def generate_alternative(
        self,
        problem: str,
        current_files: Dict[str, str],
        test_results: TestResults,
        attempt: int
    ) -> Optional[AlternativeArchitecture]:
        """Generate alternative architecture using LLM."""
        current_code = "\n\n".join([f"{name}:\n{content}" for name, content in current_files.items()])
        failing_tests = ', '.join(test_results.failed_tests[:5])
        
        prompt = f"""You are an expert Python architect. The current solution is stuck on edge cases that require a DIFFERENT ARCHITECTURAL APPROACH.

                    Use systematic reasoning to analyze the problem and propose alternative architectures:

                    STEP 1 - UNDERSTAND THE PROBLEM DOMAIN:
                    Analyze the problem statement to identify:
                    - What type of problem is this? (e.g., reactive system, API, data processing, algorithm, etc.)
                    - What are the core requirements and constraints?
                    - What patterns or paradigms are most relevant?

                    STEP 2 - ANALYZE WHY CURRENT APPROACH IS FAILING:
                    Examine the test failures to understand:
                    - What is the root cause of the failures? (not just symptoms)
                    - Why can't the current architecture handle this case?
                    - What fundamental assumption or design choice is problematic?

                    STEP 3 - GENERATE ALTERNATIVE ARCHITECTURES:
                    Propose 3 COMPLETELY DIFFERENT architectural approaches that could solve this problem.
                    For each alternative, explain:
                    - Architecture name and core concept
                    - Key design principles
                    - Why this approach addresses the root cause
                    - Trade-offs and considerations

                    This is attempt #{attempt}, so focus on approaches that are FUNDAMENTALLY DIFFERENT from typical solutions.

                    STEP 4 - SELECT AND IMPLEMENT:
                    Choose the most promising approach (approach #{min(attempt, 6)}) from your alternatives.

                    **SELECTION CRITERIA (prioritize in this order):**
                    1. **Directly addresses the failing test's edge case** (highest priority)
                    2. **Simple to implement** (fewer moving parts = fewer bugs)
                    3. **Minimal changes to working code** (preserve what already passes)
                       - {test_results.passed}/{test_results.total} tests already pass

                    Provide a complete implementation with:
                    - Clear explanation of the architecture
                    - How it solves the specific failing test cases
                    - Complete, working code

                    Problem Statement:
                    {problem}

                    Current Implementation (STUCK - Attempt #{attempt}):
                    {current_code}

                    Persistent Test Failures:
                    Failing tests: {failing_tests}
                    Passed: {test_results.passed}/{test_results.total}

                    NOW, COMPLETE THE ANALYSIS AND IMPLEMENTATION:

                    Follow the 4-step process above to:
                    1. Identify the problem domain
                    2. Analyze the failure pattern
                    3. Generate 3 architectural alternatives
                    4. Select and implement approach #{min(attempt, 6)}

                    CRITICAL REQUIREMENTS:
                    - The current architecture CANNOT fix these failures through incremental changes
                    - You MUST redesign with a FUNDAMENTALLY DIFFERENT approach
                    - Do NOT try to patch the old code - start FRESH with new design
                    - Make sure the new architecture handles the edge case that's failing
                    - Provide COMPLETE working code, not just snippets

                    OUTPUT FORMAT:
                    First, show your reasoning:
                    **ARCHITECTURE NAME:** [Give it a descriptive name]
                    **REASONING:** [Explain why this architecture solves the problem]

                    Then provide the complete code:
                    filename.py
                    ```python
                    <complete code>
                    ```

                    Generate your alternative architecture:"""
        
        try:
            response = call_llm(
                [{"role": "user", "content": prompt}],
                model=REASONING_MODEL,
                temperature=0.8  # Higher temperature for creativity
            )
            
            # Extract architecture name
            arch_name = f"Alternative #{attempt}"
            if "**ARCHITECTURE NAME:**" in response:
                name_line = response.split("**ARCHITECTURE NAME:**")[1].split("\n")[0]
                arch_name = name_line.strip()
            
            # Parse files
            files = parse_file_blocks(response)
            if not files:
                return None
            
            # Add architecture name to files dict
            files['__architecture_name__'] = arch_name
            
            return AlternativeArchitecture(
                name=arch_name,
                files=files,
                score=0
            )
        except Exception as e:
            print(f"[ERROR] Alternative architecture generation failed: {e}")
            return None

class LocalFileManager(IFileManager):
    """Concrete implementation for local file operations."""
    
    def write_files(self, files: Dict[str, str]) -> List[str]:
        """Write files to disk using helper function."""
        print(f"[DEBUG] Writing {len(files)} files: {list(files.keys())}")
        written = write_files_helper(files)
        print(f"[DEBUG] Successfully wrote: {written}")
        return written
    
    def read_files(self, patterns: List[str]) -> Dict[str, str]:
        """Read files matching patterns."""
        files = {}
        for pattern in patterns:
            for path in Path('.').rglob(pattern):
                if path.is_file():
                    try:
                        files[str(path)] = path.read_text()
                    except Exception:
                        pass
        return files

# ============================================================================
# Architecture Similarity Check (helper)
# ============================================================================

def are_architectures_similar_helper(name1: str, name2: str) -> bool:
    """Check if two architecture names are similar."""
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())
    common_words = words1 & words2
    
    if len(common_words) >= 2:
        return True
    
    return False

# ============================================================================
# Test Management
# ============================================================================

class TestManager:
    """Manages test execution and result parsing."""
    
    def __init__(self, test_runner: ITestRunner):
        self.runner = test_runner
    
    def run_and_parse(self) -> Tuple[TestResults, str]:
        """Run tests and parse results."""
        success, output = self.runner.run_tests()
        
        # Debug: Show first 1000 chars of test output to see collection
        if output:
            print(f"[DEBUG] Test output preview ({len(output)} chars total):")
            print(output[:1000])
        else:
            print("[DEBUG] Test output is empty!")
        
        results = self.runner.parse_results(output)
        
        # Debug: Show parsed results
        print(f"[DEBUG] Parsed: {results.passed} passed, {results.failed} failed, {results.total} total")
        if results.total == 0:
            print("[DEBUG] ⚠️ No tests collected! Check if test file exists and has test_ functions")
        
        return results, output
    
    def print_summary(self, results: TestResults, prefix: str = "[TESTS] "):
        """Print comprehensive test summary."""
        print(f"{prefix}{results.passed}/{results.total} passed, {results.failed} failed")
        
        if results.passed_tests:
            print(f"{prefix}✓ Passed: {', '.join(results.passed_tests[:5])}" + 
                  (f" (+{len(results.passed_tests)-5} more)" if len(results.passed_tests) > 5 else ""))
        
        if results.failed_tests:
            print(f"{prefix}✗ Failed: {', '.join(results.failed_tests[:5])}" + 
                  (f" (+{len(results.failed_tests)-5} more)" if len(results.failed_tests) > 5 else ""))
        
        # Show sample error details
        if results.error_details:
            print(f"[ERRORS] Sample failures:")
            for i, error in enumerate(results.error_details[:3]):
                print(f"  {i+1}. {error['test']}")
                for line in error['error'].split('\n')[:2]:
                    if line.strip():
                        print(f"     {line.strip()}")
    
    def print_failure_details(self, results: TestResults, prefix: str = ""):
        """Print detailed failure information."""
        if results.failed_tests:
            print(f"{prefix}✗ Failed: {', '.join(results.failed_tests[:3])}" + 
                  (f" (+{len(results.failed_tests)-3} more)" if len(results.failed_tests) > 3 else ""))
            
            if results.error_details:
                num_errors = min(3, len(results.error_details))
                for idx in range(num_errors):
                    error = results.error_details[idx]
                    print(f"{prefix}Error {idx+1}/{len(results.error_details)}: {error['test']}")
                    for line in error['error'].split('\n')[:2]:
                        if line.strip():
                            print(f"{prefix}   {line.strip()}")

# ============================================================================
# Fix Management
# ============================================================================

class FixManager:
    """Manages iterative fixing process."""
    
    def __init__(self, code_generator: ICodeGenerator, file_manager: IFileManager):
        self.generator = code_generator
        self.files = file_manager
    
    def apply_fix(
        self, 
        problem: str, 
        test_output: str, 
        solution_files: Dict[str, str]
    ) -> bool:
        """Apply fix to solution files. Returns True if successful."""
        fixed = self.generator.fix_failures(problem, test_output, solution_files)
        
        if fixed:
            solution_files.update(fixed)
            self.files.write_files(fixed)
            return True
        
        return False
    
    def refine_iteratively(
        self,
        problem: str,
        solution_files: Dict[str, str],
        test_manager: TestManager,
        max_iterations: int,
        prefix: str = "",
        initial_failed: Optional[Set[str]] = None
    ) -> Tuple[bool, TestResults, int]:
        """
        Iteratively refine solution.
        Returns: (success, final_results, stuck_count)
        """
        stuck_count = 0
        prev_failed = initial_failed if initial_failed is not None else set()
        
        for iter_num in range(max_iterations):
            # Run tests
            results, output = test_manager.run_and_parse()
            print(f"{prefix}Iteration {iter_num+1}/{max_iterations}: {results.passed}/{results.total} passed")
            
            # Show failure details
            if results.failed_tests:
                test_manager.print_failure_details(results, prefix)
            
            # Check if all tests passed
            if results.success:
                print(f"{prefix}✓ All tests passed!")
                return True, results, stuck_count
            
            # Check if stuck
            curr_failed = set(results.failed_tests)
            if curr_failed == prev_failed and len(curr_failed) > 0:
                stuck_count += 1
                if stuck_count >= 2:
                    print(f"{prefix}Stuck on same {len(curr_failed)} test(s), stopping early")
                    return False, results, stuck_count
            else:
                stuck_count = 0
            prev_failed = curr_failed
            
            # Try to fix
            print(f"{prefix}Analyzing failures and generating fix...")
            if not self.apply_fix(problem, output, solution_files):
                print(f"{prefix}Could not generate fix")
                return False, results, stuck_count
        
        # Final test
        results, _ = test_manager.run_and_parse()
        return results.success, results, stuck_count

# ============================================================================
# Architecture Management
# ============================================================================

class ArchitectureManager:
    """Manages alternative architecture exploration."""
    
    def __init__(
        self, 
        arch_generator: IArchitectureGenerator,
        file_manager: IFileManager
    ):
        self.generator = arch_generator
        self.files = file_manager
        self.tried_architectures: Dict[str, int] = {}
    
    def is_duplicate(self, arch_name: str) -> bool:
        """Check if architecture is similar to already tried ones."""
        for tried_name in self.tried_architectures.keys():
            if self._are_similar(arch_name, tried_name):
                return True
        return False
    
    def _are_similar(self, name1: str, name2: str) -> bool:
        """Check if two architecture names are semantically similar."""
        return are_architectures_similar_helper(name1, name2)
    
    def try_alternative(
        self,
        problem: str,
        current_files: Dict[str, str],
        test_results: TestResults,
        test_manager: TestManager,
        fix_manager: FixManager,
        attempt: int,
        alt_iterations: int,
        best_score: int
    ) -> Tuple[bool, Optional[AlternativeArchitecture], TestResults]:
        """
        Try an alternative architecture.
        Returns: (improved, architecture, final_results)
        """
        # Generate alternative
        alternative = self.generator.generate_alternative(
            problem, current_files, test_results, attempt
        )
        
        if not alternative:
            return False, None, test_results
        
        # Check for duplicates
        if self.is_duplicate(alternative.name):
            print(f"[ALTERNATIVE #{attempt}] ⚠️ Similar to existing architecture, skipping...")
            return False, None, test_results
        
        print(f"[ALTERNATIVE #{attempt}] Architecture: {alternative.name}")
        print(f"[ALTERNATIVE #{attempt}] Generated new architecture, iterating to improve...")
        
        # Apply alternative
        current_files.update(alternative.files)
        self.files.write_files(alternative.files)
        
        # Refine alternative
        print(f"[ALTERNATIVE #{attempt}] Running {alt_iterations} refinement iterations...")
        success, final_results, _ = fix_manager.refine_iteratively(
            problem,
            current_files,
            test_manager,
            alt_iterations,
            prefix=f"[ALTERNATIVE #{attempt}] ",
            initial_failed=set()
        )
        
        # Record attempt
        self.tried_architectures[alternative.name] = final_results.passed
        alternative.score = final_results.passed
        
        # Check if improved
        improved = final_results.passed > best_score
        
        return improved, alternative, final_results
    
    def print_summary(self):
        """Print summary of tried architectures."""
        if self.tried_architectures:
            print(f"\n[SUMMARY] Tried {len(self.tried_architectures)} alternative architecture(s):")
            for arch_name, score in self.tried_architectures.items():
                print(f"  - {arch_name}: {score} tests passed")

# ============================================================================
# Main Refinement Loop
# ============================================================================

class RefinementLoop:
    """Main test-driven refinement loop."""
    
    def __init__(
        self,
        test_manager: TestManager,
        fix_manager: FixManager,
        arch_manager: ArchitectureManager,
        config: RefinementConfig
    ):
        self.test_manager = test_manager
        self.fix_manager = fix_manager
        self.arch_manager = arch_manager
        self.config = config
    
    def run(
        self,
        problem: str,
        solution_files: Dict[str, str],
        start_time: float,
        initial_baseline: Optional[TestResults] = None
    ) -> bool:
        """
        Run test-driven refinement loop.
        Returns: True if all tests pass
        """
        # Initialize tracking
        previous_failed = set(initial_baseline.failed_tests) if initial_baseline else set()
        stuck_count = 0
        best_score = initial_baseline.passed if initial_baseline else 0
        iterations_without_improvement = 0
        
        for iteration in range(self.config.max_iterations):
            # Check timeout
            if time.time() - start_time > self.config.timeout - 60:
                print("[TIMEOUT] Stopping refinement")
                break
            
            print(f"\n--- Iteration {iteration + 1}/{self.config.max_iterations} ---")
            
            # Run tests
            results, output = self.test_manager.run_and_parse()
            self.test_manager.print_summary(results)
            
            if results.success:
                print("[SUCCESS] All tests passed!")
                return True
            
            # Track progress
            if results.passed > best_score:
                best_score = results.passed
                iterations_without_improvement = 0
                print(f"[PROGRESS] ✓ Improved! New best: {best_score}/{results.total}")
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement >= 3:
                    print(f"[PROGRESS] ⚠️ No improvement for {iterations_without_improvement} iterations")
            
            # Check if stuck (same tests failing)
            current_failed = set(results.failed_tests)
            if current_failed == previous_failed and len(current_failed) > 0:
                stuck_count += 1
                
                # Stop if stuck for too long (alternative architectures handled at round level)
                if stuck_count >= self.config.stuck_threshold:
                    print(f"\n[STUCK] Same {len(current_failed)} test(s) failing for {stuck_count} iterations")
                    print("[STUCK] Stopping refinement. Will try new architectures in next round.")
                    break
            else:
                stuck_count = 0
            
            previous_failed = current_failed
            
            # Apply fix
            print("[FIXING] Analyzing failures and generating fix...")
            if not self.fix_manager.apply_fix(problem, output, solution_files):
                print("[FIXING] Could not generate fix (network may be down)")
        
        # Print summary
        self.arch_manager.print_summary()
        
        return False

# ============================================================================
# Agent Facade
# ============================================================================

class TestDrivenAgent:
    """Main agent facade - simplified interface."""
    
    def __init__(
        self,
        test_runner: ITestRunner,
        code_generator: ICodeGenerator,
        arch_generator: IArchitectureGenerator,
        file_manager: IFileManager,
        config: Optional[RefinementConfig] = None
    ):
        self.test_manager = TestManager(test_runner)
        self.fix_manager = FixManager(code_generator, file_manager)
        self.arch_manager = ArchitectureManager(arch_generator, file_manager)
        self.code_generator = code_generator
        self.file_manager = file_manager
        self.config = config or RefinementConfig()
        self.generated_test_files=[]
        self.test_case_summary = ""  # Store test case summary for Round 1
    
    def _summarize_tests(self, test_content: str) -> str:
        """Create a concise summary of test cases.
        
        Extracts test function names and sample assertions for use in prompts.
        """
        if not test_content.strip():
            return ""
        
        summary_parts = []
        lines = test_content.split('\n')
        
        # Extract test function signatures
        for line in lines:
            stripped = line.strip()
            # Detect test functions
            if stripped.startswith('def test_'):
                summary_parts.append(f"  {stripped}")
            # Capture first assertion in each test (for context)
            elif 'assert' in stripped.lower() and len(summary_parts) > 0:
                if not any('assert' in s for s in summary_parts[-3:]):  # Only first assertion per test
                    summary_parts.append(f"    {stripped[:80]}...")  # Truncate long assertions
        
        return '\n'.join(summary_parts[:30])  # Limit to 30 lines
    
    def solve_create(self, problem: str, timeout: int) -> str:
        """Solve CREATE mode problem."""
        start_time = time.time()

        code_skeleton = get_code_skeleton()
        
        print("\n" + "="*80)
        print("CREATE MODE - Test-Driven Development")
        if ENABLE_PARALLEL:
            print(" PARALLEL MODE ENABLED")
        print("="*80)
        
        # Check for existing tests first
        print("\n[STEP 1] Checking for existing test files...")
        existing_tests = list(Path('.').glob('*test*.py'))
        
        if not existing_tests:
            logger.info("[STEP 1] No existing tests found, generating test suite...")
            
            # Generate tests using OOP approach
            # Don't pass skeleton - let LLM determine the module name from problem statement
            # The skeleton is just a template, not the actual solution module
            test_files = self.code_generator.generate_tests(problem, {})
            
            if test_files:
                # Write test files
                self.file_manager.write_files(test_files)
                self.generated_test_files.extend(test_files.keys())
                logger.info(f"[STEP 1] Created {len(test_files)} test file(s): {list(test_files.keys())}")
                
                # Create summary of test cases for Round 1
                test_content = "\n\n".join([f"{name}\n{content}" for name, content in test_files.items()])
                self.test_case_summary = self._summarize_tests(test_content)
                print("\n[STEP 1] Test case summary created:")
                print("=" * 80)
                print(self.test_case_summary[:500] + "..." if len(self.test_case_summary) > 500 else self.test_case_summary)
                print("=" * 80)
            else:
                logger.warning("[STEP 1] Failed to generate test files")
                self.test_case_summary = ""
        else:
            logger.info(f"[STEP 1] Found {len(existing_tests)} existing test files")
            logger.info("[STEP 1] Using existing tests from dataset")
            
            # Read existing test files and create summary
            existing_test_content = ""
            for test_file in existing_tests:
                with open(test_file, 'r') as f:
                    existing_test_content += f"{test_file.name}\n{f.read()}\n\n"
            self.test_case_summary = self._summarize_tests(existing_test_content)
            print("\n[STEP 1] Existing test case summary created")
        
        # Step 2: Generate solution(s) with restart mechanism to escape local minima
        if ENABLE_PARALLEL:
            # Parallel mode with refinement in each round
            print("\n[STEP 2] Multi-round parallel generation with refinement...")
            
            max_attempts = 2  # Try up to 2 full attempts with fresh context to escape local minima
            max_rounds = 3  # Increased from 2 - more rounds = more diverse architectures
            optimal_workers = ResourceManager.get_optimal_workers()
            solutions_per_round = optimal_workers  # Start with N workers, dynamic queue adds more as they complete
            
            best_across_all_attempts = None
            solution_files = {}
            found_perfect = False
            
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    print(f"\n{'#'*80}")
                    print(f" ATTEMPT {attempt}/{max_attempts} - RESTARTING WITH FRESH CONTEXT")
                    print(f"{'#'*80}")
                    print(f"[RESTART] Previous attempt achieved {best_across_all_attempts.score if best_across_all_attempts else 0} tests")
                    print(f"[RESTART] Starting over with clean slate to escape local minima...")
                
                # Create fresh instances for this attempt (new random seed, fresh LLM context)
                parallel_gen = ParallelSolutionGenerator(
                    self.code_generator,
                    self.test_manager,
                    self.file_manager,
                    test_case_summary=self.test_case_summary  # Pass test summary
                )
                
                refinement_loop = RefinementLoop(
                    self.test_manager,
                    self.fix_manager,
                    self.arch_manager,
                    self.config
                )
                
                best_overall = None
                found_perfect = False
                stuck_rounds = 0  # Track consecutive rounds with same score
                previous_score = -1
                
                # Track best candidate from previous round to learn from its failures
                best_from_previous_round = None
                
                for round_num in range(1, max_rounds + 1):
                    print(f"\n{'='*80}")
                    print(f"ROUND {round_num}/{max_rounds} - Generate N Solutions + Refine Best")
                    print(f"{'='*80}")
                    
                    # Generate N solutions in parallel with different architectures
                    # Pass best candidate info from previous round to target its failures
                    print(f"\n[ROUND {round_num}] Generating {solutions_per_round} solutions in parallel...")
                    candidates = parallel_gen.generate_multiple_solutions(
                        problem, 
                        solutions_per_round, 
                        best_candidate_info=best_from_previous_round
                    )
                    
                    if not candidates:
                        print(f"[ROUND {round_num}] No valid solutions generated")
                        continue
                    
                    # DEBUG: Check what we got back
                    print(f"\n[DEBUG] Received {len(candidates)} candidates from parallel generation")
                    for i, c in enumerate(candidates[:3]):  # Show top 3
                        print(f"[DEBUG]   {i+1}. {c.architecture_name[:60]}... - {c.score}/{c.test_results.total if c.test_results else 0} (is_perfect={c.is_perfect})")
                    
                    # Pick best from this round
                    best_candidate = candidates[0]  # Already sorted by score
                    print(f"\n[ROUND {round_num}] Best candidate: '{best_candidate.architecture_name}' "
                          f"({best_candidate.score}/{best_candidate.test_results.total if best_candidate.test_results else 0} tests)")
                    print(f"[DEBUG] Best candidate is_perfect: {best_candidate.is_perfect}")
                    
                    # Store solution files for patch generation
                    solution_files = best_candidate.solution_files
                    
                    # Check if already perfect
                    if best_candidate.is_perfect:
                        print(f"\n[ROUND {round_num}] Perfect solution found!")
                        self.file_manager.write_files(solution_files)
                        found_perfect = True
                        break
                    
                    # Try refining ALL candidates (or top N) to maximize chance of finding perfect solution
                    # Filter candidates that are worth refining (e.g., at least 50% tests passing)
                    total_tests = best_candidate.test_results.total if best_candidate.test_results else 14
                    min_score_for_refinement = total_tests // 2  # At least 50% passing
                    
                    refinement_candidates = [c for c in candidates if c.score >= min_score_for_refinement]
                    print(f"\n[ROUND {round_num}] Trying refinement on {len(refinement_candidates)} candidates (score >= {min_score_for_refinement})...")
                    
                    for idx, candidate in enumerate(refinement_candidates, 1):
                        print(f"\n[ROUND {round_num}] Refining candidate {idx}/{len(refinement_candidates)}: '{candidate.architecture_name[:60]}...' ({candidate.score}/{total_tests} tests)")
                        self.file_manager.write_files(candidate.solution_files)
                        
                        success = refinement_loop.run(problem, candidate.solution_files, start_time, candidate.test_results)
                        
                        if success:
                            print(f"\n[ROUND {round_num}] Refinement achieved perfect solution on candidate {idx}!")
                            solution_files = candidate.solution_files
                            found_perfect = True
                            break
                    
                    if found_perfect:
                        break
                    
                    # Track best overall
                    final_results, _ = self.test_manager.run_and_parse()
                    current_score = final_results.passed
                    
                    # Check if stuck in local minimum (same score as previous round)
                    if current_score == previous_score and current_score < final_results.total:
                        stuck_rounds += 1
                        print(f"[LOCAL MINIMUM DETECTION] Same score ({current_score}) for {stuck_rounds} consecutive rounds")
                        
                        # Early restart if stuck for 2 rounds
                        if stuck_rounds >= 2:
                            print(f"\n[EARLY RESTART] Stuck at {current_score}/{final_results.total} for 2 rounds. Breaking to restart...")
                            break
                    else:
                        stuck_rounds = 0
                    
                    previous_score = current_score
                    
                    if best_overall is None or final_results.passed > best_overall.score:
                        best_overall = SolutionCandidate(
                            solution_files=solution_files.copy(),
                            test_results=final_results,
                            architecture_name=best_candidate.architecture_name,
                            generation_time=0
                        )
                        print(f"[ROUND {round_num}] New best overall: {final_results.passed}/{final_results.total} tests")
                    
                    # Prepare info for next round - focus on best candidate's failures
                    if round_num < max_rounds and final_results.failed > 0:
                        best_from_previous_round = {
                            'architecture': best_candidate.architecture_name,
                            'test_results': final_results,
                            'failed_tests': final_results.failed_tests,
                            'error_details': final_results.error_details
                        }
                        print(f"\n[ROUND {round_num}] No perfect solution yet. Next round will target these {final_results.failed} failures...")
                    elif round_num < max_rounds:
                        print(f"\n[ROUND {round_num}] No perfect solution yet. Trying NEW architectures in next round...")
                
                # End of rounds for this attempt
                # Track best across all attempts
                if best_overall:
                    if best_across_all_attempts is None or best_overall.score > best_across_all_attempts.score:
                        best_across_all_attempts = best_overall
                        print(f"\n[ATTEMPT {attempt}] New best across all attempts: {best_overall.score} tests")
                
                # If found perfect, break out of attempt loop
                if found_perfect:
                    print(f"\n[SUCCESS] Perfect solution found in attempt {attempt}!")
                    break
                
                # If this is the last attempt, use best solution
                if attempt == max_attempts:
                    print(f"\n[FINAL] All {max_attempts} attempts completed. Best: {best_across_all_attempts.score if best_across_all_attempts else 0} tests")
            
            # Use best solution from all attempts
            if not found_perfect and best_across_all_attempts:
                print(f"\n[FINAL] Using best solution from all attempts: {best_across_all_attempts.score} tests passed")
                self.file_manager.write_files(best_across_all_attempts.solution_files)
                solution_files = best_across_all_attempts.solution_files
            elif not found_perfect:
                print("[WARNING] No valid solution found across all attempts!")
        else:
            # Sequential mode: Original behavior
            print("\n[STEP 2] Generating initial solution...")
            solution_files = self.code_generator.generate_solution(problem)
            created_files = self.file_manager.write_files(solution_files)
            print(f"[STEP 2] Created {len(created_files)} solution files")
            
            # Run baseline
            print("[STEP 2] Running initial tests to establish baseline...")
            initial_results, _ = self.test_manager.run_and_parse()
            print(f"[STEP 2] Baseline: {initial_results.passed}/{initial_results.total} tests passing")
            
            # Refine
            print("\n[STEP 3] Test-driven refinement...")
            refinement_loop = RefinementLoop(
                self.test_manager,
                self.fix_manager,
                self.arch_manager,
                self.config
            )
            refinement_loop.run(problem, solution_files, start_time, initial_results)
        
        # Generate patch
        print("\n[COMPLETE] Generating patch...")
        if not solution_files:
            print("[ERROR] No solution files generated - cannot create patch")
            return ""
        patch = self.get_final_git_patch()
        print(f"[COMPLETE] Generated patch ({len(patch)} bytes)")
        return patch
    
    def solve_fix(self, problem: str, timeout: int) -> str:
        """Solve FIX mode problem."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("FIX MODE - Iterative Debugging")
        print("="*80)
        
        # Step 1: Find relevant files
        print("\n[STEP 1] Finding relevant files...")
        relevant_files = self.file_manager.read_files(['*.py'])
        print(f"[STEP 1] Found {len(relevant_files)} relevant files")
        
        # Step 2: Generate reproduction test
        print("\n[STEP 2] Generating reproduction test...")
        test_file = self.code_generator.generate_tests(problem, relevant_files)
        if test_file:
            self.file_manager.write_files(test_file)
            # self.generated_test_files.extend(test_file.keys())
            print("[STEP 2] Created reproduction test")
        
        # Step 3: Refine
        print("\n[STEP 3] Iterative fixing...")
        config = RefinementConfig(max_alternatives=5)  # Fewer alternatives for FIX
        refinement_loop = RefinementLoop(
            self.test_manager,
            self.fix_manager,
            self.arch_manager,
            config
        )
        
        refinement_loop.run(problem, relevant_files, start_time, None)
        
        # Generate patch
        print("\n[COMPLETE] Generating patch...")
        patch = self.get_final_git_patch()
        print(f"[COMPLETE] Generated patch ({len(patch)} bytes)")
        
        return patch
    
    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume. THREAD-SAFE: Uses explicit working directory.
        """
        try:
            # Use explicit working directory - never rely on CWD!
            work_dir = get_main_work_dir()
            logger.info(f"[PATCH] Generating patch in {work_dir}")
            
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(self, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p, work_dir))
            except Exception:
                pass

            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                cwd=work_dir,  # EXPLICIT!
                capture_output=True, 
                text=True, 
                timeout=30, 
                check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                logger.info(f"[PATCH] Staging {len(to_add)} files")
                subprocess.run(
                    ["git", "add", "--"] + to_add, 
                    cwd=work_dir,  # EXPLICIT!
                    check=True, 
                    timeout=30
                )
            else:
                logger.warning("[PATCH] No files to stage! Modified files: %s, Excluded: %s", ls, exclude)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                cwd=work_dir,  # EXPLICIT!
                capture_output=True, 
                text=True, 
                timeout=30, 
                check=True
            )

            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            logger.info(f"[PATCH] Generated {len(patch_text)} byte patch")
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"

# ============================================================================
# Helper Functions for Thread-Safe Operations
# ============================================================================

def cleanup_temp_dir_with_retry(temp_dir: str, max_attempts: int = 3):
    """Cleanup temp directory with retry for locked files."""
    import shutil
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(temp_dir, ignore_errors=(attempt < max_attempts - 1))
            logger.debug(f"[CLEANUP] Removed {temp_dir}")
            return
        except (PermissionError, OSError) as e:
            if attempt < max_attempts - 1:
                time.sleep(0.2 * (attempt + 1))  # Exponential backoff
            else:
                logger.warning(f"[CLEANUP] Could not remove {temp_dir}: {e}")

# ============================================================================
# Parallel Solution Generator
# ============================================================================

class ParallelSolutionGenerator:
    """Generates and tests multiple solutions in parallel with COT-based architecture diversity."""
    
    def __init__(self, code_generator: ICodeGenerator, test_manager: TestManager, file_manager: IFileManager, test_case_summary: str = ""):
        self.code_generator = code_generator
        self.test_manager = test_manager
        self.file_manager = file_manager
        self.arch_manager = LLMArchitectureGenerator()  # Used for dynamic architecture generation
        self.num_workers = ResourceManager.get_optimal_workers()
        self.perfect_solution_found = False
        self.termination_lock = Lock()  # Protect the flag
        self.used_architecture_descriptions = []  # Track used architectures
        self.architecture_lock = Lock()  # Protect architecture tracking
        self.test_case_summary = test_case_summary  # Store test case summary for prompts
        
        # CRITICAL: Store main working directory and test files BEFORE threading
        self.main_work_dir = get_main_work_dir()
        
        # Discover test files once
        import glob
        self.test_files = []
        for pattern in ["test*.py", "*_test.py"]:
            self.test_files.extend(
                glob.glob(os.path.join(self.main_work_dir, pattern))
            )
        
        logger.info(f"[PARALLEL] Main work dir: {self.main_work_dir}")
        logger.info(f"[PARALLEL] Found {len(self.test_files)} test files")
    
    def generate_diverse_architectures(self, problem: str, num_architectures: int, best_candidate_info: Optional[Dict] = None) -> List[str]:
        """Use COT to generate N diverse architectural approaches for the problem.
        
        Args:
            problem: Problem statement
            num_architectures: Number of architectures to generate
            best_candidate_info: Dict with 'architecture', 'test_results', 'failed_tests', 'error_details' from best previous candidate
        """
        problem_summary = truncate_text(problem, max_chars=2000)
        
        # Build list of previously used architectures with their results
        avoid_list = ""
        if self.used_architecture_descriptions:
            avoid_list = "\n\nPREVIOUSLY TRIED ARCHITECTURES (DO NOT REPEAT):\n"
            for i, arch in enumerate(self.used_architecture_descriptions[-6:], 1):
                avoid_list += f"{i}. {arch}\n"
        
        # Add detailed failure analysis from best previous candidate
        failure_analysis = ""
        if best_candidate_info:
            test_results = best_candidate_info.get('test_results')
            failed_tests = best_candidate_info.get('failed_tests', [])
            error_details = best_candidate_info.get('error_details', [])
            
            if test_results:
                failure_analysis = "\n\n🎯 CRITICAL - LEARN FROM BEST PREVIOUS ATTEMPT:\n"
                failure_analysis += f"Architecture: {best_candidate_info.get('architecture', 'Unknown')}\n"
                failure_analysis += f"Score: {test_results.passed}/{test_results.total} tests passed\n"
                
                if failed_tests:
                    failure_analysis += f"\n❌ FAILED TESTS ({len(failed_tests)}):\n"
                    # Show ALL failed tests (not just 3) since we need complete context
                    for i, test_name in enumerate(failed_tests, 1):
                        failure_analysis += f"{i}. {test_name}\n"
                
                if error_details:
                    failure_analysis += "\n🔍 ERROR DETAILS (what went wrong):\n"
                    # Show ALL errors with MORE detail (increased from 200 to 800 chars)
                    for i, error in enumerate(error_details, 1):
                        error_msg = truncate_text(str(error), max_chars=800)
                        failure_analysis += f"{i}. {error_msg}\n"
                    
                    failure_analysis += "\n💡 YOUR TASK:\n"
                    failure_analysis += f"Generate {num_architectures} NEW architectures that specifically ADDRESS these failures.\n"
                    failure_analysis += "Think about what architectural patterns would PREVENT these specific errors.\n"
                else:
                    failure_analysis += "\n💡 Close to perfect! Generate architectures that handle edge cases better.\n"
        
        prompt = f"""You are an expert software architect. Generate {num_architectures} COMPLETELY DIFFERENT architectural approaches to solve this problem.

            Problem Summary:
            {problem_summary}
            {avoid_list}
            {failure_analysis}
            
            Use Chain-of-Thought reasoning to ensure diversity:
            
            STEP 1 - ANALYZE PROBLEM DOMAIN:
            - What type of problem is this? (algorithm, data processing, API, state management, etc.)
            - What are the core operations needed?
            - What are the key constraints?
            
            STEP 2 - BRAINSTORM DIVERSE PARADIGMS:
            Think of {num_architectures} FUNDAMENTALLY DIFFERENT ways to approach this:
            - Different programming paradigms (functional, OOP, procedural, data-driven)
            - Different design patterns (strategy, state machine, builder, pipeline, etc.)
            - Different data structures (dict-based, class-based, list-based, generator-based)
            - Different control flows (recursive, iterative, event-driven, declarative)
            
            STEP 3 - SELECT {num_architectures} MOST DISTINCT APPROACHES:
            Choose approaches that are MAXIMALLY DIFFERENT from each other and from previously used ones.
            
            CRITICAL REQUIREMENTS:
            - Each architecture must be FUNDAMENTALLY different (not just minor variations)
            - Avoid any similarity to previously used architectures
            - Be specific about the core design principle of each approach
            
            OUTPUT FORMAT - CRITICAL:
            You MUST output EXACTLY {num_architectures} lines in this format (no preamble, no explanation):
            
            Architecture 1: [One concise sentence describing the core architectural approach]
            Architecture 2: [One concise sentence describing a COMPLETELY DIFFERENT approach]
            Architecture 3: [One concise sentence describing yet ANOTHER DIFFERENT approach]
            ...
            
            DO NOT include any introductory text. Start directly with "Architecture 1:".
            
            Generate {num_architectures} diverse architectures NOW:"""
        
        try:
            response = call_llm(
                [{"role": "user", "content": prompt}],
                model=REASONING_MODEL,
                temperature=0.9  # High temperature for diversity
            )
            
            # Parse architectures from response - be strict first, then flexible
            architectures = []
            lines = response.split('\n')
            
            # First pass: Look for proper "Architecture N:" format
            for line in lines:
                line = line.strip()
                if line.startswith('Architecture ') and ':' in line:
                    arch_desc = line.split(':', 1)[1].strip()
                    if arch_desc and len(arch_desc) > 20:  # Meaningful description
                        architectures.append(arch_desc)
            
            # Second pass: Try numbered list format "1.", "2.", etc.
            if len(architectures) < num_architectures:
                for line in lines:
                    line = line.strip()
                    # Match "1. Description" or "Approach 1: Description"
                    if any(line.startswith(f"{i}.") for i in range(1, 11)) or line.startswith('Approach '):
                        if ':' in line:
                            arch_desc = line.split(':', 1)[1].strip()
                        else:
                            arch_desc = line.split('.', 1)[1].strip() if '.' in line else line
                        if arch_desc and len(arch_desc) > 20 and arch_desc not in architectures:
                            architectures.append(arch_desc)
            
            # Fallback: Extract meaningful sentences (skip preamble)
            if len(architectures) < num_architectures:
                print(f"[COT] Warning: Only parsed {len(architectures)} architectures, using fallback")
                skip_keywords = ['here are', 'fundamentally different', 'architectural approaches', 
                                'step 1', 'step 2', 'step 3', 'critical', 'output format']
                for line in lines:
                    line = line.strip()
                    # Skip preamble and instructions
                    if any(keyword in line.lower() for keyword in skip_keywords):
                        continue
                    # Skip lines that are too short or look like headers
                    if len(line) > 40 and not line.startswith('#') and not line.startswith('**'):
                        if line not in architectures:
                            architectures.append(line)
                            if len(architectures) >= num_architectures:
                                break
            
            # Track these architectures
            with self.architecture_lock:
                self.used_architecture_descriptions.extend(architectures)
            
            print(f"[COT] Generated {len(architectures)} diverse architectures:")
            for i, arch in enumerate(architectures, 1):
                print(f"  {i}. {arch[:80]}...")
            
            return architectures[:num_architectures]
            
        except Exception as e:
            print(f"[COT] Failed to generate architectures: {e}")
            # Fallback: return generic hints with high temperature
            return [f"Approach {i+1}" for i in range(num_architectures)]
    
    def _build_failure_context(self, candidates: List[SolutionCandidate]) -> Optional[Dict]:
        """Build failure context with PRIORITY on best candidate's errors.
        
        Strategy: Always include ALL errors from best candidate, then add errors from others
        to fill context budget. This prevents best candidate errors from being truncated.
        """
        if not candidates:
            return None
        
        # Find best candidate first (highest priority)
        best = max(candidates, key=lambda c: c.score)
        
        # Collect all unique failed tests across all candidates
        all_failed_tests = set()
        
        # PRIORITY 1: Best candidate's errors (NEVER truncated)
        best_errors = []
        if best.test_results:
            if best.test_results.failed > 0:
                all_failed_tests.update(best.test_results.failed_tests)
                if hasattr(best.test_results, 'error_details'):
                    best_errors = best.test_results.error_details  # ALL errors from best
        
        # PRIORITY 2: Other candidates' errors (truncated to conserve context)
        other_errors = []
        other_candidates = [c for c in candidates if c != best]
        for candidate in other_candidates:
            if candidate.test_results and candidate.test_results.failed > 0:
                all_failed_tests.update(candidate.test_results.failed_tests)
                if hasattr(candidate.test_results, 'error_details'):
                    # Only take 1 error per other candidate (heavily truncated)
                    other_errors.extend(candidate.test_results.error_details[:1])
        
        if not all_failed_tests:
            return None
        
        # Combine: ALL best errors + limited other errors (max 3 from others)
        combined_errors = best_errors + other_errors[:3]
        
        return {
            'architecture': best.architecture_name,
            'test_results': best.test_results,
            'failed_tests': list(all_failed_tests),
            'error_details': combined_errors,  # Best errors ALWAYS included in full
            'num_completed': len(candidates),
            'best_error_count': len(best_errors),  # For debugging
            'other_error_count': len(other_errors[:3])
        }
    
    def _build_failure_breakdown(self, context: Dict) -> str:
        """Build detailed failure breakdown text for code generator with prioritized errors."""
        if not context:
            return ""
        
        breakdown = "\n\n🎯 FAILURE ANALYSIS FROM PREVIOUS ATTEMPTS:\n"
        breakdown += f"Completed solutions: {context.get('num_completed', 0)}\n"
        breakdown += f"Best candidate: {context.get('architecture', 'Unknown')}\n"
        
        # Show error prioritization info
        best_err_count = context.get('best_error_count', 0)
        other_err_count = context.get('other_error_count', 0)
        if best_err_count > 0:
            breakdown += f"📊 Errors: {best_err_count} from best candidate (full detail), {other_err_count} from others (summarized)\n"
        
        failed_tests = context.get('failed_tests', [])
        if failed_tests:
            breakdown += f"\n❌ FAILED TESTS ({len(failed_tests)}):\n"
            # Show all failed tests (important context)
            for i, test in enumerate(failed_tests, 1):
                breakdown += f"  {i}. {test}\n"
        
        errors = context.get('error_details', [])
        if errors:
            breakdown += f"\n🔍 DETAILED ERROR ANALYSIS ({len(errors)} errors):\n"
            # Show all errors with more detail (prioritized: best candidate errors come first)
            for i, error in enumerate(errors, 1):
                error_str = str(error)[:500]  # Increased from 150 to 500 chars
                marker = "⭐" if i <= best_err_count else "•"  # Mark best candidate errors
                breakdown += f"  {marker} Error {i}: {error_str}\n"
        
        breakdown += "\n💡 CRITICAL: Focus on the ⭐ starred errors first (from best candidate).\n"
        breakdown += "These are the remaining issues preventing a perfect solution.\n"
        return breakdown
    
    def generate_and_test_solution(self, problem: str, architecture_hint: Optional[str] = None, failure_hints: Optional[str] = None) -> Optional[SolutionCandidate]:
        """Generate a single solution and test it in an isolated directory.
        
        THREAD-SAFE: Each execution uses isolated temp directory.
        NEVER uses os.chdir() - all operations use explicit paths.
        """
        # Check early termination
        with self.termination_lock:
            if self.perfect_solution_found:
                logger.info(f"[PARALLEL] Skipping '{architecture_hint or 'default'}' - perfect solution found")
                return None
        
        start_time = time.time()
        temp_dir = None
        arch_name = architecture_hint or "default"
        
        try:
            # Step 1: Generate solution code (no I/O yet)
            # Include test case summary in the problem statement for Round 1
            enhanced_problem = problem
            if self.test_case_summary:
                enhanced_problem = f"{problem}\n\n{'='*80}\nGENERATED TEST CASES SUMMARY:\n{'='*80}\n{self.test_case_summary}\n{'='*80}\n"
            
            solution_files = self.code_generator.generate_solution(
                enhanced_problem, architecture_hint, failure_hints
            )
            
            if not solution_files:
                logger.warning(f"[PARALLEL] No solution files generated for '{arch_name}'")
                return SolutionCandidate(
                    solution_files={},
                    test_results=None,
                    architecture_name=arch_name,
                    generation_time=time.time() - start_time
                )
            
            # Check early termination before expensive testing
            with self.termination_lock:
                if self.perfect_solution_found:
                    return None
            
            # Step 2: Create isolated temp directory (atomic)
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp(prefix=f"sol_{arch_name[:20]}_", dir="/tmp")
            logger.debug(f"[PARALLEL] Created temp dir: {temp_dir}")
            
            # Step 3: Copy test files to temp directory
            for test_file in self.test_files:
                try:
                    shutil.copy2(test_file, temp_dir)
                except Exception as e:
                    logger.warning(f"[PARALLEL] Could not copy {test_file}: {e}")
            
            # Step 4: Write solution files to temp directory (EXPLICIT paths)
            for filename, content in solution_files.items():
                filepath = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Step 5: Run tests with EXPLICIT working directory (no chdir!)
            success, output = run_tests_helper(work_dir=temp_dir, timeout=30)
            
            # Step 6: Parse results
            test_results = self.test_manager.runner.parse_results(output)
            
            generation_time = time.time() - start_time
            
            candidate = SolutionCandidate(
                solution_files=solution_files,
                test_results=test_results,
                architecture_name=arch_name,
                generation_time=generation_time
            )
            
            # Step 7: Signal early termination if perfect
            if candidate.is_perfect:
                with self.termination_lock:
                    self.perfect_solution_found = True
                logger.info(f"[PARALLEL] ⚡ Perfect solution: '{arch_name}'")
            
            return candidate
            
        except Exception as e:
            logger.error(f"[PARALLEL] Error in '{arch_name}': {e}")
            import traceback
            logger.debug(f"[PARALLEL] Traceback: {traceback.format_exc()}")
            
            return SolutionCandidate(
                solution_files={},
                test_results=None,
                architecture_name=arch_name,
                generation_time=time.time() - start_time
            )
        
        finally:
            # Step 8: Cleanup temp directory (with retry)
            if temp_dir and os.path.exists(temp_dir):
                cleanup_temp_dir_with_retry(temp_dir, max_attempts=3)
    
    def generate_multiple_solutions(
        self, 
        problem: str, 
        num_solutions: int = 3,
        architecture_hints: Optional[List[str]] = None,
        best_candidate_info: Optional[Dict] = None
    ) -> List[SolutionCandidate]:
        """Generate multiple solutions in parallel with GUARANTEED diverse architectures.
        
        Args:
            problem: Problem statement
            num_solutions: Number of solutions to generate
            architecture_hints: Pre-generated architecture hints (optional)
            best_candidate_info: Info about best previous candidate (architecture, test_results, failed_tests, errors)
        """
        # Reset futures counter for this round
        self._futures_after_perfect = 0
        
        # CRITICAL: Generate diverse architectures BEFORE threading to guarantee uniqueness
        if not architecture_hints:
            print(f"\n[COT] Generating up to {num_solutions} diverse architectures using Chain-of-Thought...")
            architecture_hints = self.generate_diverse_architectures(problem, num_solutions, best_candidate_info)
            
            # Accept whatever architectures are available - don't force padding
            if not architecture_hints:
                print("[COT] ⚠️ No architectures generated, using single default approach")
                architecture_hints = ["Default approach"]
            elif len(architecture_hints) < num_solutions:
                actual_count = len(architecture_hints)
                print(f"[COT] ✓ Generated {actual_count} unique architectures (requested {num_solutions})")
                print("[COT] This is the maximum diversity available for this problem - adjusting expectations")
                # Update num_solutions to match what's actually available
                num_solutions = actual_count
            else:
                print(f"[COT] ✓ Successfully generated {len(architecture_hints)} diverse architectures")
        
        print(f"\n[PARALLEL] Generating {num_solutions} solutions in parallel with {self.num_workers} workers...")
        print("[PARALLEL] Each thread will use a UNIQUE architecture pattern")
        print("[PARALLEL] 🚀 Dynamic queue enabled - will generate new architectures as solutions complete")
        
        # Build failure hints from best_candidate_info (CRITICAL: pass to solution generator!)
        failure_hints = None
        if best_candidate_info:
            failure_hints = self._build_failure_context(best_candidate_info)
            if failure_hints:
                print("[PARALLEL] 📋 Passing failure context from best candidate to ALL solution generators")
        
        candidates = []
        use_dynamic_queue = True  # Enable dynamic architecture generation
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit initial tasks WITH failure_hints
            future_to_hint = {
                executor.submit(self.generate_and_test_solution, problem, hint, failure_hints): hint
                for hint in architecture_hints[:num_solutions]
            }
            
            completed_count = 0
            all_futures = set(future_to_hint.keys())  # Track ALL futures including dynamic ones
            
            # Collect results as they complete
            # CRITICAL: We need to re-check as_completed() when new futures are added
            while all_futures:
                # Get next completed future
                done_futures = []
                try:
                    for future in as_completed(all_futures, timeout=0.1):
                        done_futures.append(future)
                        break  # Process one at a time to allow dynamic queue to add more
                except TimeoutError:
                    # No futures completed within timeout, continue waiting
                    continue
                
                if not done_futures:
                    continue  # No futures completed yet, check again
                
                future = done_futures[0]
                all_futures.remove(future)  # Remove completed future from tracking
                hint = future_to_hint[future]
                try:
                    candidate = future.result()
                    
                    # Skip None results (early termination)
                    if candidate is None:
                        completed_count += 1
                        continue
                    
                    candidates.append(candidate)
                    completed_count += 1
                    
                    if candidate.test_results:
                        print(f"[PARALLEL] Solution '{candidate.architecture_name}': "
                              f"{candidate.test_results.passed}/{candidate.test_results.total} tests passed "
                              f"({candidate.generation_time:.1f}s)")
                        print(f"[DEBUG] Candidate is_perfect={candidate.is_perfect}, success={candidate.test_results.success}, passed={candidate.test_results.passed}, failed={candidate.test_results.failed}")
                        if candidate.is_perfect:
                            print(f"[DEBUG] ✅ Perfect candidate added to list! Total candidates: {len(candidates)}")
                    else:
                        print(f"[PARALLEL] Solution '{candidate.architecture_name}': failed to generate")
                    
                    # Check if ANY candidate in the list is perfect (not just the current one)
                    # This handles the case where a perfect solution was added earlier but another thread completed after
                    perfect_candidates = [c for c in candidates if c.is_perfect]
                    print(f"[DEBUG] Checking for perfect solutions: {len(perfect_candidates)} found out of {len(candidates)} candidates")
                    
                    # CRITICAL: Only break if we actually HAVE a perfect candidate in our list
                    # The self.perfect_solution_found flag might be set by a thread that hasn't
                    # been yielded by as_completed() yet!
                    if perfect_candidates:
                        print("[PARALLEL] ⚡ Perfect solution found - returning immediately!")
                        print(f"[DEBUG] Perfect candidate: {perfect_candidates[0].architecture_name} ({perfect_candidates[0].score}/{perfect_candidates[0].test_results.total if perfect_candidates[0].test_results else 0})")
                        
                        # Cancel all pending futures to stop unnecessary work
                        pending_count = 0
                        running_count = 0
                        for f in future_to_hint.keys():
                            if not f.done():
                                if f.cancel():  # Returns True if successfully cancelled
                                    pending_count += 1
                                else:
                                    running_count += 1  # Already running, can't cancel
                        
                        if pending_count > 0:
                            print(f"[PARALLEL] Cancelled {pending_count} pending tasks")
                        if running_count > 0:
                            print(f"[PARALLEL] {running_count} tasks still running (will be ignored)")
                        
                        # Sort to put perfect solution first (is_perfect=True, then by score)
                        candidates.sort(key=lambda c: (not c.is_perfect, -c.score))
                        print(f"[DEBUG] After sort, candidates[0]: {candidates[0].architecture_name} ({candidates[0].score}/{candidates[0].test_results.total if candidates[0].test_results else 0}, is_perfect={candidates[0].is_perfect})")
                        
                        # Shutdown executor without waiting for running threads
                        print("[PARALLEL] Shutting down executor without waiting for running threads...")
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            # Python < 3.9 doesn't support cancel_futures parameter
                            executor.shutdown(wait=False)
                        
                        # Break out of the loop
                        break
                    elif self.perfect_solution_found:
                        # Perfect solution found by another thread but not yet in candidates list
                        # Wait for at most 2 more futures to collect the perfect solution, then force exit
                        print("[DEBUG] Perfect solution flag set but not in candidates yet, will process at most 2 more futures...")
                        max_futures_after_perfect = 2
                        futures_processed_after_perfect = getattr(self, '_futures_after_perfect', 0)
                        self._futures_after_perfect = futures_processed_after_perfect + 1
                        
                        if self._futures_after_perfect > max_futures_after_perfect:
                            print(f"[PARALLEL] ⚠️ Processed {self._futures_after_perfect} futures after perfect flag set, but perfect candidate not in list")
                            print("[PARALLEL] Force exiting with best available candidate to prevent hang")
                            # Sort and return best candidate we have
                            if candidates:
                                candidates.sort(key=lambda c: (not c.is_perfect, -c.score))
                                try:
                                    executor.shutdown(wait=False, cancel_futures=True)
                                except TypeError:
                                    executor.shutdown(wait=False)
                                break
                            else:
                                print("[PARALLEL] ERROR: No candidates available, continuing...")
                                # Reset counter and continue
                                self._futures_after_perfect = 0
                    
                    # Dynamic queue: Generate new architecture and submit new task
                    if use_dynamic_queue and not self.perfect_solution_found:
                        # Build failure context from ALL completed solutions
                        failed_tests_context = self._build_failure_context(candidates)
                        
                        if failed_tests_context:
                            num_failed = len(failed_tests_context.get('failed_tests', []))
                            num_completed = failed_tests_context.get('num_completed', len(candidates))
                            
                            # Generate ONE new architecture based on ALL failures
                            print(f"[DYNAMIC] 🔄 Generating new architecture based on {num_completed} completed solutions...")
                            print(f"[DYNAMIC]    Targeting {num_failed} unique failed tests from all solutions")
                            
                            new_archs = self.generate_diverse_architectures(
                                problem, 
                                num_architectures=1,
                                best_candidate_info=failed_tests_context
                            )
                            
                            if new_archs:
                                new_hint = new_archs[0]
                                print(f"[DYNAMIC] ✓ Submitting new task: {new_hint[:80]}...")
                                print(f"[DYNAMIC]    Passing failure context to solution generator")
                                
                                # Build detailed failure breakdown for code generator
                                failure_breakdown_text = self._build_failure_breakdown(failed_tests_context)
                                
                                # Submit new task to the executor
                                new_future = executor.submit(
                                    self.generate_and_test_solution, 
                                    problem, 
                                    new_hint,  # architecture hint
                                    failure_breakdown_text  # failure hints
                                )
                                future_to_hint[new_future] = new_hint
                                all_futures.add(new_future)  # Track the new future
                                print(f"[DEBUG] Added dynamic future to tracking. Total futures: {len(all_futures)}")
                        
                except Exception as e:
                    print(f"[PARALLEL] Solution with hint '{hint}' failed: {e}")
                    completed_count += 1
        
        # Sort by is_perfect first, then by score (best first)
        candidates.sort(key=lambda c: (not c.is_perfect, -c.score))
        
        return candidates
    
    def find_best_solution(
        self,
        problem: str,
        max_rounds: int = 3,
        solutions_per_round: int = 3
    ) -> Optional[SolutionCandidate]:
        """
        Generate solutions in multiple rounds until a perfect solution is found.
        
        Strategy:
        - Round 1: Generate N solutions with default architectures
        - Round 2+: If no perfect solution, generate N more with different architectures
        - Return best solution found
        """
        
        all_candidates = []
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n[PARALLEL] === Round {round_num}/{max_rounds} ===")
            
            # Reset termination flag for new round
            with self.termination_lock:
                self.perfect_solution_found = False
            
            # Generate architecture hints for this round
            if round_num == 1:
                hints = [None] * solutions_per_round  # Default architectures
            else:
                # Generate different architecture hints based on previous failures
                hints = [f"alternative_{i}_round_{round_num}" for i in range(solutions_per_round)]
            
            # Generate and test solutions in parallel
            candidates = self.generate_multiple_solutions(problem, solutions_per_round, hints)
            all_candidates.extend(candidates)
            
            # Check if we found a perfect solution (early termination worked!)
            perfect_solutions = [c for c in candidates if c.is_perfect]
            if perfect_solutions:
                best = perfect_solutions[0]
                print(f"\n[PARALLEL] ✓ Found perfect solution: '{best.architecture_name}' "
                      f"({best.test_results.passed}/{best.test_results.total} tests)")
                return best
            
            # Show best so far
            if candidates:
                best_this_round = candidates[0]
                print(f"[PARALLEL] Best this round: '{best_this_round.architecture_name}' "
                      f"({best_this_round.score} tests passed)")
        
        # No perfect solution found, return best overall
        if all_candidates:
            all_candidates.sort(key=lambda c: (not c.is_perfect, -c.score))
            best = all_candidates[0]
            print(f"\n[PARALLEL] No perfect solution found. Best: '{best.architecture_name}' "
                  f"({best.score} tests passed)")
            return best
        
        return None

# ============================================================================
# Entry Point - Independent Function
# ============================================================================

def agent_main(input_data: Dict[str, Any]) -> str:
    """
    Independent entry point for the agent.
    Called by benchmark system with problem data.
    
    Args:
        input_data: Dict with keys:
            - problem_statement: str
            - timeout: int (optional, default 1800)
            - mode: str (optional, "create" or "fix", auto-detected if None)
    
    Returns:
        patch: str (git diff format)
    """
    # Extract input
    problem_statement = input_data.get('problem_statement', '')
    timeout = input_data.get('timeout', 1800)
    mode = input_data.get('mode', None)
    
    logger.info(f"[AGENT] Test-Driven Agent - Mode: {mode or 'AUTO'}")
    print(f"\n{'='*80}")
    print("TEST-DRIVEN ITERATIVE AGENT")
    print("="*80)
    
    # Step 1: Setup working directory (BEFORE any threading)
    repo_dir = os.path.abspath("repo")
    sys.path.insert(0, repo_dir)

    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    
    # Step 2: Store main working directory globally (CRITICAL!)
    set_main_work_dir(os.getcwd())
    logger.info(f"[AGENT] Main work dir: {get_main_work_dir()}")
    print(f"[SETUP] Working directory: {get_main_work_dir()}")
    print(f"[SETUP] Problem type: {mode or 'AUTO-DETECT'}")

    # Step 3: Initialize git (BEFORE any threading)
    ensure_git_initialized()

    # Step 4: Auto-detect mode if not specified
    if mode is None:
        # Simple heuristic: check for existing Python files
        py_files = list(Path('.').rglob('*.py'))
        has_existing_code = len(py_files) > 0
        
        keywords_fix = ['fix', 'bug', 'error', 'issue', 'broken', 'incorrect']
        keywords_create = ['create', 'implement', 'write', 'build', 'generate']
        
        statement_lower = problem_statement.lower()
        fix_score = sum(1 for kw in keywords_fix if kw in statement_lower)
        create_score = sum(1 for kw in keywords_create if kw in statement_lower)
        
        mode = "fix" if (has_existing_code and fix_score > create_score) else "create"
    
    logger.info(f"[AGENT] Mode: {mode.upper()}")
    
    # Create concrete implementations (these would be imported from actual implementations)
    # For now, these are placeholder - you need to implement these based on your existing code
    test_runner = PytestRunner()
    code_generator = LLMCodeGenerator()
    arch_generator = LLMArchitectureGenerator()
    file_manager = LocalFileManager()
    
    # Configure based on mode
    config = RefinementConfig(
        max_iterations=10,
        stuck_threshold=2,
        timeout=timeout,
    )
    
    # Create agent
    agent = TestDrivenAgent(
        test_runner,
        code_generator,
        arch_generator,
        file_manager,
        config
    )
    
    # Solve problem based on mode
    try:
        if mode == "create":
            patch = agent.solve_create(problem_statement, timeout)
        else:
            patch = agent.solve_fix(problem_statement, timeout)
        
        # Reset git state with explicit cwd
        subprocess.run(
            ["git", "reset", "--hard"],
            cwd=get_main_work_dir(),  # EXPLICIT!
            check=False
        )
        return patch
    
    except Exception:
        logger.exception("[AGENT] Agent failed")
        import traceback
        traceback.print_exc()
        return ""

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: CREATE mode
    input_data = {
        'problem_statement': '''
        Implement a reactive system with two types of cells:
        - Input cells: Hold mutable values
        - Compute cells: Calculate based on input cells
        Cells can have callbacks that fire when values change.
        ''',
        'timeout': 1800,
        'mode': 'create'
    }
    
    patch = agent_main(input_data)
    print(f"\nGenerated patch:\n{patch}")
    
    # Example 2: FIX mode
    input_data = {
        'problem_statement': 'Fix the bug where callbacks fire even when value doesn\'t change',
        'timeout': 1800,
        'mode': 'fix'
    }
    
    patch = agent_main(input_data)
    print(f"\nGenerated patch:\n{patch}")

def get_code_skeleton() -> str:
    # Initialize the result string
    result = ""
    
    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"
    
    return result

# def generate_test_files(problem_statement: str, code_skeleton: str) -> str:
#     retry = 0
#     while retry < 10:
#         try:
#             logger.info("Starting test cases generation")
            
#             testcases = generate_testcases_with_multi_step_reasoning(problem_statement, code_skeleton)
            
#             if testcases:
#                 logger.info("Generated testcases successfully using multi-step reasoning")
#                 return testcases
#             else:
#                 logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
#                 # Fallback to original single-step approach if multi-step fails
#                 messages = [
#                     {
#                         "role": "system",
#                         "content": GENERATE_INITIAL_TESTCASES_PROMPT
#                     },
#                     {
#                         "role": "user",
#                         "content": f"""Problem Statement:\n{problem_statement}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
#                     }
#                 ]
                
#                 response = call_llm(messages, model=CODING_MODEL)
                
#                 # Clean up the response
#                 testcases = response.strip()
#                 if testcases.startswith('```python'):
#                     testcases = testcases[9:]
#                 if testcases.startswith('```'):
#                     testcases = testcases[3:]
#                 if testcases.endswith('```'):
#                     testcases = testcases[:-3]
#                 testcases = testcases.strip()
                
#                 logger.info("Generated testcases successfully using fallback approach")
#                 return testcases
            
#         except Exception as e:
#             logger.error(f"Error generating initial solution: {str(e)}")
#             retry += 1
#             time.sleep(2)
    
#     if retry >= 10:
#         logger.error("Failed to generate initial solution")
#         return ""
#     return ""


# # ============================================================================
# # Test Case Generation - Helper Functions
# # ============================================================================

# # Constants for test generation
# MAX_RETRY_ATTEMPTS = 10
# NUM_TEST_GENERATIONS = 5
# TEST_GENERATION_TEMPERATURE = 0.3
# RETRY_DELAY_SECONDS = 2

# FILE_NAME_REQUIREMENT_MSG = (
#     "Include file name in the response. example:\n"
#     "```python\n"
#     "test_a.py\n"
#     "contents of test_a.py\n\n"
#     "test_b.py\n"
#     "contents of test_b.py\n"
#     "```"
# )


# def _extract_test_function_names(testcode: str) -> Set[str]:
#     """Extract function names from test code to create a signature for comparison.
    
#     Args:
#         testcode: Test code string to parse
        
#     Returns:
#         Set of test function names found in the code
#     """
#     function_names = set()
#     test_function_patterns = [
#         r'def\s+(test_\w+)',    # def test_something
#         r'def\s+(test\w+)',     # def testSomething
#         r'def\s+(\w*test\w*)',  # any function containing 'test'
#     ]
    
#     for pattern in test_function_patterns:
#         matches = re.findall(pattern, testcode, re.IGNORECASE)
#         function_names.update(matches)
    
#     return function_names


# def _clean_markdown_code_block(response: str) -> str:
#     """Remove markdown code block formatting from LLM response.
    
#     Args:
#         response: Raw LLM response potentially wrapped in markdown
        
#     Returns:
#         Cleaned code string without markdown markers
#     """
#     cleaned = response.strip()
#     if cleaned.startswith('```python'):
#         cleaned = cleaned[9:]
#     elif cleaned.startswith('```'):
#         cleaned = cleaned[3:]
#     if cleaned.endswith('```'):
#         cleaned = cleaned[:-3]
#     return cleaned.strip()


# def _create_test_generation_messages(problem_statement: str, code_skeleton: str) -> List[Dict[str, str]]:
#     """Create initial messages for test case generation.
    
#     Args:
#         problem_statement: The problem description
#         code_skeleton: Code skeleton/template
        
#     Returns:
#         List of message dictionaries for LLM
#     """
#     return [
#         {
#             "role": "system",
#             "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Problem Statement:\n{problem_statement}\n\n"
#                 f"Code skeleton: \n{code_skeleton}\n\n"
#                 f"Generate the complete and correct testcases in python files.\n\n"
#                 f"STRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\n"
#                 f"{FILE_NAME_REQUIREMENT_MSG}"
#             )
#         }
#     ]


# def _create_test_check_messages(problem_statement: str, code_skeleton: str, testcode: str) -> List[Dict[str, str]]:
#     """Create messages for test case validation.
    
#     Args:
#         problem_statement: The problem description
#         code_skeleton: Code skeleton/template
#         testcode: Generated test code to validate
        
#     Returns:
#         List of message dictionaries for LLM
#     """
#     return [
#         {
#             "role": "system",
#             "content": TESTCASES_CHECK_PROMPT
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Problem statement: {problem_statement}\n\n"
#                 f"Code skeleton: \n{code_skeleton}\n\n"
#                 f"Generated Test Code:\n{testcode}\n\n"
#                 f"Analyze this code for invalid testcases. Return ONLY the final Python test code."
#             )
#         }
#     ]


# def _validate_test_file_format(testcases: str) -> bool:
#     """Check if the test code starts with a valid Python file name.
    
#     Args:
#         testcases: The test code string to validate
        
#     Returns:
#         True if format is valid, False otherwise
#     """
#     lines = testcases.split("\n")
#     return len(lines) > 0 and lines[0].endswith(".py")


# def _generate_single_testset(problem_statement: str, code_skeleton: str) -> Tuple[str, Set[str]]:
#     """Generate a single test set with retry logic.
    
#     Args:
#         problem_statement: The problem description
#         code_skeleton: Code skeleton/template
        
#     Returns:
#         Tuple of (testcode, function_names)
#     """
#     messages = _create_test_generation_messages(problem_statement, code_skeleton)
    
#     for attempt in range(MAX_RETRY_ATTEMPTS):
#         try:
#             # Step 1: Generate test cases
#             testcode_response = call_llm(
#                 messages, 
#                 model=CODING_MODEL, 
#                 temperature=TEST_GENERATION_TEMPERATURE
#             )
#             logger.info("Step 1 - Testcase Generation completed")
            
#             # Step 2: Validate and refine test cases
#             check_messages = _create_test_check_messages(
#                 problem_statement, 
#                 code_skeleton, 
#                 testcode_response
#             )
#             testcode_checked = call_llm(check_messages, model=CODING_MODEL)
#             logger.info("Step 2 - Testcase check completed")
            
#             # Step 3: Clean and validate format
#             testcases = _clean_markdown_code_block(testcode_checked)
            
#             if not _validate_test_file_format(testcases):
#                 logger.warning(f"Attempt {attempt + 1}: Invalid file format, retrying...")
#                 messages.append({"role": "assistant", "content": testcode_checked})
#                 messages.append({"role": "user", "content": FILE_NAME_REQUIREMENT_MSG})
#                 continue
            
#             # Step 4: Extract test signatures
#             function_names = _extract_test_function_names(testcases)
#             logger.info(f"Generated testset with functions: {function_names}")
#             return testcases, function_names
            
#         except Exception as e:
#             logger.error(f"Attempt {attempt + 1} failed: {e}")
#             if attempt < MAX_RETRY_ATTEMPTS - 1:
#                 time.sleep(RETRY_DELAY_SECONDS)
    
#     logger.error(f"Failed to generate valid test set after {MAX_RETRY_ATTEMPTS} attempts")
#     return "", set()


# def _select_most_common_testset(test_sets: List[str], function_signatures: List[Tuple[str, ...]]) -> str:
#     """Select the test set that matches the most common function signature pattern.
    
#     Args:
#         test_sets: List of generated test code strings
#         function_signatures: List of function signature tuples
        
#     Returns:
#         The selected test code string
#     """
#     signature_counts = Counter(function_signatures)
#     most_common_signature, most_common_count = signature_counts.most_common(1)[0]
    
#     logger.info(
#         f"Most common function signature: {most_common_signature} "
#         f"(appeared {most_common_count}/{len(test_sets)} times)"
#     )
    
#     # Find first test set matching the most common signature
#     for i, signature in enumerate(function_signatures):
#         if signature == most_common_signature:
#             logger.info(f"Selected test set {i + 1} as it matches the most common pattern")
#             return test_sets[i]
    
#     # Fallback: return first valid test set
#     logger.warning("No matching signature found, returning first test set")
#     return test_sets[0]


# def generate_testcases_with_multi_step_reasoning(
#     problem_statement: str, 
#     code_skeleton: str
# ) -> str:
#     """Generate test cases using multi-step reasoning with consensus validation.
    
#     This function generates multiple test sets and selects the most consistent one
#     based on function signature patterns to ensure quality and reliability.
    
#     Args:
#         problem_statement: Description of the problem to solve
#         code_skeleton: Template or skeleton code for the solution
        
#     Returns:
#         Selected test code string, or empty string if generation fails
#     """
#     logger.info(
#         f"Generating {NUM_TEST_GENERATIONS} test sets to find the most common pattern..."
#     )
    
#     test_sets = []
#     function_signatures = []
    
#     # Generate multiple test sets
#     for i in range(NUM_TEST_GENERATIONS):
#         logger.info(f"Generating test set {i + 1}/{NUM_TEST_GENERATIONS}")
#         testcode, function_names = _generate_single_testset(problem_statement, code_skeleton)
        
#         if testcode and function_names:
#             test_sets.append(testcode)
#             function_signatures.append(tuple(sorted(function_names)))
#         else:
#             logger.warning(f"Failed to generate valid test set {i + 1}")
    
#     # Validate we have at least one valid test set
#     if not test_sets:
#         logger.error("Failed to generate any valid test sets")
#         return ""
    
#     # Select the most consistent test set
#     return _select_most_common_testset(test_sets, function_signatures)

# def summarize_test_cases(test_files_content: str) -> str:
#     """Create a concise summary of generated test cases for inclusion in prompts.
    
#     Extracts:
#     - File header/comments (first 10 lines or until first import)
#     - All test function signatures and their docstrings
#     - Sample assertions from each test (first 2 assertions)
#     """
#     if not test_files_content.strip():
#         return ""
    
#     summary_parts = []
#     lines = test_files_content.split('\n')
#     current_file = None
    
#     for i, line in enumerate(lines):
#         stripped = line.strip()
        
#         # Detect file names
#         if stripped.endswith('.py') and ' ' not in stripped and len(stripped) > 3:
#             current_file = stripped
#             summary_parts.append(f"\n=== {current_file} ===")
            
#             # Extract header comments (next 10 lines or until import)
#             header_lines = []
#             for j in range(i+1, min(i+11, len(lines))):
#                 next_line = lines[j].strip()
#                 if next_line.startswith('#') or next_line.startswith('"""') or next_line.startswith("'''"):
#                     header_lines.append(lines[j])
#                 elif next_line.startswith('import') or next_line.startswith('from'):
#                     break
            
#             if header_lines:
#                 summary_parts.append('\n'.join(header_lines[:5]))  # Max 5 header lines
    
#     # Extract test function signatures and sample assertions
#     in_test_function = False
#     test_name = None
#     test_docstring = None
#     assertion_count = 0
#     max_assertions_per_test = 2
    
#     for line in lines:
#         stripped = line.strip()
        
#         # Detect test function
#         if stripped.startswith('def test_'):
#             # Save previous test if any
#             if test_name:
#                 summary_parts.append('')  # Blank line between tests
            
#             in_test_function = True
#             test_name = stripped
#             assertion_count = 0
#             summary_parts.append(f"\n    {test_name}")
#             continue
        
#         if in_test_function:
#             # Capture docstring
#             if ('"""' in stripped or "'''" in stripped) and test_docstring is None:
#                 test_docstring = stripped
#                 summary_parts.append(f"        {test_docstring}")
#                 continue
            
#             # Capture assertions (limit to first 2 per test)
#             if assertion_count < max_assertions_per_test:
#                 if 'assert' in stripped.lower() or 'self.assert' in line:
#                     summary_parts.append(f"        {stripped}")
#                     assertion_count += 1
            
#             # End of test function (next function or class)
#             if stripped.startswith('def ') and not stripped.startswith('def test_'):
#                 in_test_function = False
#             elif stripped.startswith('class '):
#                 in_test_function = False
    
#     return '\n'.join(summary_parts)

# def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
#     import os
    
#     created_files = []
    
#     if not initial_solution.strip():
#         print("No solution content to process")
#         return created_files
    
#     lines = initial_solution.split('\n')
#     current_filename = None
#     current_content = []
    
#     for line in lines:
#         # Check if this line is just a Python filename (*.py pattern)
#         stripped_line = line.strip()
        
#         # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
#         if (stripped_line.endswith('.py') and 
#             ' ' not in stripped_line and 
#             len(stripped_line) > 3 and 
#             '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
#             not stripped_line.startswith('#')):
#             if current_filename and current_content:
#                 file_path = os.path.join(base_dir, current_filename)
#                 os.makedirs(os.path.dirname(file_path), exist_ok=True)
#                 content = '\n'.join(current_content).strip()
#                 with open(file_path, 'w', encoding='utf-8') as f:
#                     f.write(content)
#                 created_files.append(file_path)
#             current_filename = stripped_line
#             current_content = []
#         else:

#             if current_filename:  # Only collect content if we have a filename
#                 current_content.append(line)
#     if current_filename and current_content:
#         file_path = os.path.join(base_dir, current_filename)
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         content = '\n'.join(current_content).strip()
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(content)
#         created_files.append(file_path)
#         print(f"Created file: {file_path}")
#     return created_files
