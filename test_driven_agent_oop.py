"""
Test-Driven Agent - Object-Oriented Architecture

SOLID Principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Easy to extend with new strategies
- Liskov Substitution: Interfaces can be swapped
- Interface Segregation: Focused interfaces
- Dependency Inversion: Depend on abstractions

Sandbox Compatibility:
- Uses SANDBOX_PROXY_URL for LLM API calls
- Uses os.getloadavg() instead of psutil for resource management
- Configured with RUN_ID for tracking in sandbox environment
- Includes proper logging setup for sandbox monitoring
"""

import time
import os
import subprocess
import requests
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    max_iterations: int = 8  # Keep low - rely on diverse architectures instead
    stuck_threshold: int = 2  # Keep low - quickly move to next architecture
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
    def generate_solution(self, problem: str, architecture_hint: Optional[str] = None) -> Dict[str, str]:
        """Generate initial solution with optional architecture hint for diversity."""
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

def detect_expected_module_name(test_file_path: str = "tests.py") -> str:
    """Detect what module name the test file expects by parsing imports.
    
    Returns the expected module name (e.g., 'main', 'beer_song', etc.)
    Defaults to 'main' if no test file or unclear imports.
    """
    if not os.path.exists(test_file_path):
        return "main"
    
    try:
        with open(test_file_path, 'r') as f:
            content = f.read()
        
        # Look for imports like "from MODULE import" or "import MODULE"
        import_patterns = [
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',  # from module import ...
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',           # import module
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for module in matches:
                # Exclude standard library and common test modules
                if module not in ['unittest', 'pytest', 'sys', 'os', 'json', 're', 'math', 'typing']:
                    print(f"[DETECT] Test file expects module: '{module}'")
                    return module
        
        # Default to 'main' if no clear module found
        print(f"[DETECT] No clear module found in tests, defaulting to 'main'")
        return "main"
    
    except Exception as e:
        print(f"[DETECT] Error reading test file: {e}, defaulting to 'main'")
        return "main"

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

def run_tests_helper(test_file: Optional[str] = None, timeout: int = 30) -> Tuple[bool, str]:
    """Run tests and return (success, output)."""
    try:
        test_files_found = []
        for f in Path('.').glob('*.py'):
            if f.name.startswith('test_') or f.name == 'tests.py':
                test_files_found.append(str(f))
        
        if not test_files_found and not test_file:
            return False, "[TEST ERROR] No test files found"
        
        if test_file:
            cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
        else:
            if test_files_found:
                cmd = ["python", "-m", "pytest"] + test_files_found + ["-v", "--tb=short"]
            else:
                cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=os.getcwd())
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
            for line in failure_section.split('\n')[:200]:  # Limit to avoid huge logs
                if line.startswith('_'):  # Test separator like "_ test_name _"
                    # Save previous test's error
                    if current_test and error_lines:
                        results['error_details'].append({
                            'test': current_test,
                            'error': '\n'.join(error_lines[:10])  # First 10 lines of error
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
                    'error': '\n'.join(error_lines[:10])
                })
    
    # Debug output
    if results['total'] > 0:
        print(f"[PARSE] Found {results['passed']} passed, {results['failed']} failed, {results['total']} total tests")
    
    return results

def get_git_diff_helper() -> str:
    """Generate git diff patch that can always be applied."""
    try:
        # Stage all Python files
        subprocess.run("git add *.py", shell=True, capture_output=True, cwd=".")
        
        # Generate diff from the initial commit (skeleton) to current STAGED state
        # Use --cached to show staged changes (after git add)
        result = subprocess.run(
            ["git", "diff", "--cached", "HEAD", "--relative", "--binary"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="."
        )
        
        patch = result.stdout
        
        # If no diff (files unchanged), generate a patch from scratch
        if not patch:
            # Get all Python files and create a patch manually
            py_files = list(Path('.').glob('*.py'))
            if py_files:
                # Create a unified diff format patch
                patch_lines = []
                for py_file in py_files:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    # Create a patch that adds the entire file
                    lines = content.split('\n')
                    patch_lines.append(f"diff --git a/{py_file.name} b/{py_file.name}")
                    patch_lines.append(f"new file mode 100644")
                    patch_lines.append(f"index 0000000..{hash(content) % 10000000:07x}")
                    patch_lines.append(f"--- /dev/null")
                    patch_lines.append(f"+++ b/{py_file.name}")
                    patch_lines.append(f"@@ -0,0 +1,{len(lines)} @@")
                    for line in lines:
                        patch_lines.append(f"+{line.rstrip()}")
                    patch_lines.append("")
                
                patch = '\n'.join(patch_lines)
        
        # Strip trailing whitespace from each line to avoid git warnings
        if patch:
            lines = patch.split('\n')
            cleaned_lines = [line.rstrip() for line in lines]
            patch = '\n'.join(cleaned_lines)
        
        return patch
    except Exception as e:
        print(f"[GIT] Error getting diff: {e}")
        import traceback
        traceback.print_exc()
        return ""

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
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def generate_solution(self, problem: str, architecture_hint: Optional[str] = None) -> Dict[str, str]:
        """Generate solution using LLM with optional architecture hint for diversity."""
        # Truncate problem statement to avoid context overflow
        problem = truncate_text(problem, max_chars=5000)
        
        # Detect expected module name from tests
        expected_module = detect_expected_module_name("tests.py")
        expected_filename = f"{expected_module}.py"
        print(f"[CODE_GEN] Generating solution for module: {expected_filename}")
        
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
        
        prompt = f"""You are an expert Python developer. Use step-by-step reasoning to solve this problem.

                    Problem Statement:
                    {problem}
                    {test_examples}
                    {architecture_guidance}
                    
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
                    
                    CRITICAL: Your solution MUST be in a file named '{expected_filename}' (the tests import from this module).
                    
                    Now generate the complete solution in this format:
                    {expected_filename}
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
            
            # Ensure the file has the correct expected name
            if expected_filename not in files:
                # Rename the first file to the expected name
                if files:
                    first_key = list(files.keys())[0]
                    content = files[first_key]
                    files = {expected_filename: content}
                    print(f"[CODE_GEN] ⚠️ Renamed '{first_key}' to '{expected_filename}' to match test imports")
            
            return files
        except Exception as e:
            print(f"[ERROR] Solution generation failed: {e}")
            return {}
    
    def generate_tests(self, problem: str, solution: Dict[str, str]) -> Dict[str, str]:
        """Generate tests using LLM with multi-step validation."""
        solution_summary = "\n\n".join([f"{name}:\n{content[:500]}..." for name, content in solution.items()])
        
        # Step 1: Generate initial tests
        prompt = f"""You are an expert test developer. Generate comprehensive tests for this solution.

                    Problem Statement:
                    {problem}

                    Solution Files:
                    {solution_summary}

                    Important things:
                    1. Test functions declared in code skeleton, don't customized those prototypes.
                    2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
                    3. Do not generate testcases that are not mentioned in problem statement
                    4. Minimize all testcases as you have context and generation limit

                    Requirements:
                    1. Use pytest framework
                    2. Test all functions and edge cases
                    3. Include boundary conditions, empty inputs, invalid inputs
                    4. Output in format: test_filename.py followed by test code
                    5. Import from solution files correctly

                    Example format:
                    test_main.py
                    import pytest
                    from main import solution

                    def test_basic():
                        assert solution() == expected

                    def test_edge_case():
                        assert solution(edge_input) == expected

                    Generate comprehensive tests now:"""
        
        try:
            # Step 1: Generate tests
            print("[TEST_GEN] Step 1: Generating initial tests...")
            response = call_llm(
                [{"role": "user", "content": prompt}],
                model=CODING_MODEL
            )
            
            # Step 2: Validate and refine tests
            print("[TEST_GEN] Step 2: Validating and refining tests...")
            validation_prompt = f"""You are an expert test reviewer. Analyze the generated tests for validity.

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

                If tests are valid and complete:
                - Return the original tests unchanged

                STRICT REQUIREMENT: Return ONLY the final Python test code with file names.

                Example format:
                test_main.py
                import pytest
                from main import solution

                def test_basic():
                    assert solution() == expected
                """
            
            validated_response = call_llm(
                [{"role": "user", "content": validation_prompt}],
                model=CODING_MODEL,
                temperature=0.0  # Deterministic for validation
            )
            
            print("[TEST_GEN] Step 3: Parsing validated tests...")
            files = parse_file_blocks(validated_response)
            
            if not files:
                print("[TEST_GEN] Validation failed, using original tests...")
                files = parse_file_blocks(response)
            else:
                print("[TEST_GEN] ✓ Tests validated and refined")
            
            # Debug: Show what test files were parsed
            if files:
                print(f"[TEST_GEN] Parsed test files: {list(files.keys())}")
                
                # CRITICAL FIX: Rename any file to 'tests.py' to avoid overwriting solution
                # The LLM sometimes returns 'main.py' or other names
                if 'tests.py' not in files:
                    # Take the first file and rename it to tests.py
                    first_key = list(files.keys())[0]
                    test_content = files[first_key]
                    files = {'tests.py': test_content}
                    print(f"[TEST_GEN] ⚠️ Renamed '{first_key}' to 'tests.py' to avoid conflicts")
            else:
                print("[TEST_GEN] ✗ No test files parsed!")
                print(f"[TEST_GEN] Response preview: {validated_response[:500]}")
            
            return files if files else {}
        except Exception as e:
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
        # Detect expected module name
        expected_module = detect_expected_module_name("tests.py")
        expected_filename = f"{expected_module}.py"
        
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
            
            # Ensure the file has the correct expected name
            if fixed_files and expected_filename not in fixed_files:
                # Rename the first file to the expected name
                first_key = list(fixed_files.keys())[0]
                content = fixed_files[first_key]
                fixed_files = {expected_filename: content}
                print(f"[FIX] ⚠️ Renamed '{first_key}' to '{expected_filename}' to match test imports")
            
            return fixed_files if fixed_files else None
        except Exception as e:
            print(f"[ERROR] Fix generation failed: {e}")
            return None

class LLMArchitectureGenerator(IArchitectureGenerator):
    """Concrete implementation for generating alternative architectures."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
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
    
    def solve_create(self, problem: str, timeout: int) -> str:
        """Solve CREATE mode problem."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("CREATE MODE - Test-Driven Development")
        if ENABLE_PARALLEL:
            print("🚀 PARALLEL MODE ENABLED")
        print("="*80)
        
        # Check for existing tests first
        print("\n[STEP 1] Checking for existing test files...")
        existing_tests = list(Path('.').glob('*test*.py'))
        
        if not existing_tests:
            print("[STEP 1] No existing tests found, generating test suite...")
            test_files = self.code_generator.generate_tests(problem, {})
            self.file_manager.write_files(test_files)
            print("[STEP 1] Created test files")
        else:
            print(f"[STEP 1] Found {len(existing_tests)} existing test files")
            print("[STEP 1] Using existing tests from dataset")
        
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
                    print(f"🔄 ATTEMPT {attempt}/{max_attempts} - RESTARTING WITH FRESH CONTEXT")
                    print(f"{'#'*80}")
                    print(f"[RESTART] Previous attempt achieved {best_across_all_attempts.score if best_across_all_attempts else 0} tests")
                    print(f"[RESTART] Starting over with clean slate to escape local minima...")
                
                # Create fresh instances for this attempt (new random seed, fresh LLM context)
                parallel_gen = ParallelSolutionGenerator(
                    self.code_generator,
                    self.test_manager,
                    self.file_manager
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
                        print(f"[ROUND {round_num}] ✗ No valid solutions generated")
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
                        print(f"\n[ROUND {round_num}] ✓ Perfect solution found!")
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
                            print(f"\n[ROUND {round_num}] ✓ Refinement achieved perfect solution on candidate {idx}!")
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
        patch = self._generate_patch(solution_files)
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
        
        # Detect if this is a SWEBench problem (large repo)
        is_swebench = len(relevant_files) > 100
        
        if is_swebench:
            print(f"[STEP 2] 🔍 Detected SWEBench problem ({len(relevant_files)} files)")
            print("[STEP 2] Setting up proper test environment for SWEBench...")
            
            # For SWEBench, we need to:
            # 1. Use the repo's existing test infrastructure
            # 2. Run tests in a proper venv with dependencies
            # 3. This is similar to how benchmark does it
            
            # The test_manager should already be configured to use the repo's tests
            # We just need to make sure we're running in the right environment
            
            # Try to run initial tests to see baseline
            print("[STEP 2] Running baseline tests...")
            initial_results, _ = self.test_manager.run_and_parse()
            print(f"[STEP 2] Baseline: {initial_results.passed}/{initial_results.total} tests passing")
            
            # Step 3: Iterative fixing with proper test feedback
            print("\n[STEP 3] Iterative fixing with test feedback...")
            config = RefinementConfig(max_iterations=8, stuck_threshold=3)  # More iterations for SWEBench
            refinement_loop = RefinementLoop(
                self.test_manager,
                self.fix_manager,
                self.arch_manager,
                config
            )
            
            # Run refinement with initial test results as feedback
            refinement_loop.run(problem, relevant_files, start_time, initial_results)
            
        else:
            # Small problem (Polyglot) - original flow
            print("\n[STEP 2] Checking for existing tests...")
            import glob
            existing_tests = glob.glob("test*.py") + glob.glob("*_test.py")
            if existing_tests:
                print(f"[STEP 2] Using existing test file: {existing_tests[0]}")
            else:
                print("[STEP 2] ⚠️ No test file found - generating reproduction test...")
                test_file = self.code_generator.generate_tests(problem, relevant_files)
                if test_file:
                    self.file_manager.write_files(test_file)
                    print("[STEP 2] Created reproduction test")
            
            # Step 3: Refine
            print("\n[STEP 3] Iterative fixing...")
            config = RefinementConfig(max_iterations=5, stuck_threshold=2)
            refinement_loop = RefinementLoop(
                self.test_manager,
                self.fix_manager,
                self.arch_manager,
                config
            )
            
            refinement_loop.run(problem, relevant_files, start_time, None)
        
        # Generate patch
        print("\n[COMPLETE] Generating patch...")
        patch = self._generate_patch(relevant_files)
        print(f"[COMPLETE] Generated patch ({len(patch)} bytes)")
        
        return patch
    
    def _generate_patch(self, files: Dict[str, str]) -> str:
        """Generate git diff patch using helper function."""
        return get_git_diff_helper()

# ============================================================================
# Parallel Solution Generator
# ============================================================================

class ParallelSolutionGenerator:
    """Generates and tests multiple solutions in parallel with COT-based architecture diversity."""
    
    def __init__(self, code_generator, test_manager, file_manager):
        self.code_generator = code_generator
        self.test_manager = test_manager
        self.file_manager = file_manager
        # No file_lock needed - each thread uses independent directory
        self.num_workers = ResourceManager.get_optimal_workers()
        self.perfect_solution_found = False  # Flag for early termination
        self.termination_lock = Lock()  # Protect the flag
        self.used_architecture_descriptions = []  # Track used architectures
        self.architecture_lock = Lock()  # Protect architecture tracking
    
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
                    for i, test_name in enumerate(failed_tests[:3], 1):  # Show top 3 failures
                        failure_analysis += f"{i}. {test_name}\n"
                
                if error_details:
                    failure_analysis += "\n🔍 ERROR DETAILS (what went wrong):\n"
                    for i, error in enumerate(error_details[:3], 1):  # Show top 3 errors
                        error_msg = truncate_text(str(error), max_chars=200)
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
        """Build failure context from all completed candidates for dynamic queue."""
        if not candidates:
            return None
        
        # Collect all unique failed tests across all candidates
        all_failed_tests = set()
        all_errors = []
        
        for candidate in candidates:
            if candidate.test_results and candidate.test_results.failed > 0:
                # Get failed test names
                failed_names = [name for name in candidate.test_results.failed_tests if name]
                all_failed_tests.update(failed_names)
                
                # Get error details
                if hasattr(candidate.test_results, 'error_details'):
                    all_errors.extend(candidate.test_results.error_details[:2])  # Top 2 errors per candidate
        
        if not all_failed_tests:
            return None
        
        # Find best candidate for context
        best = max(candidates, key=lambda c: c.score)
        
        return {
            'architecture': best.architecture_name,
            'test_results': best.test_results,
            'failed_tests': list(all_failed_tests),
            'error_details': all_errors[:5],  # Top 5 errors total
            'num_completed': len(candidates)
        }
    
    def _build_failure_breakdown(self, context: Dict) -> str:
        """Build detailed failure breakdown text for code generator."""
        if not context:
            return ""
        
        breakdown = "\n\n🎯 FAILURE ANALYSIS FROM PREVIOUS ATTEMPTS:\n"
        breakdown += f"Completed solutions: {context.get('num_completed', 0)}\n"
        
        failed_tests = context.get('failed_tests', [])
        if failed_tests:
            breakdown += f"\n❌ FAILED TESTS ({len(failed_tests)}):\n"
            for i, test in enumerate(failed_tests[:5], 1):
                breakdown += f"  {i}. {test}\n"
        
        errors = context.get('error_details', [])
        if errors:
            breakdown += "\n🔍 ERROR PATTERNS:\n"
            for i, error in enumerate(errors[:3], 1):
                error_str = str(error)[:150]
                breakdown += f"  {i}. {error_str}\n"
        
        breakdown += "\n💡 Focus on fixing these specific failures in your implementation.\n"
        return breakdown
    
    def generate_and_test_solution(self, problem: str, architecture_hint: Optional[str] = None, failure_hints: Optional[str] = None) -> Optional[SolutionCandidate]:
        """Generate a single solution and test it in an independent directory (fully parallel). Returns None if terminated early."""
        # Check if another thread already found a perfect solution
        with self.termination_lock:
            if self.perfect_solution_found:
                print(f"[PARALLEL] Skipping '{architecture_hint or 'default'}' - perfect solution already found")
                return None
        
        start_time = time.time()
        
        try:
            # Generate solution
            solution_files = self.code_generator.generate_solution(problem, architecture_hint)
            arch_name = architecture_hint or "default"
            
            # Check again before expensive testing
            with self.termination_lock:
                if self.perfect_solution_found:
                    print(f"[PARALLEL] Skipping test for '{arch_name}' - perfect solution already found")
                    return None
            
            # Create per-task temporary directory for parallel testing
            import tempfile
            import shutil
            
            # Create unique temp directory per task (not per thread!)
            # Use UUID to ensure uniqueness even if thread is reused by dynamic queue
            task_id = str(uuid4())[:12]
            temp_dir = f"/tmp/task_{task_id}"
            
            try:
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save current directory
                original_cwd = os.getcwd()
                
                # Copy test files to temp directory (CRITICAL!)
                import glob
                test_files = glob.glob(os.path.join(original_cwd, "test*.py")) + glob.glob(os.path.join(original_cwd, "*_test.py"))
                for test_file in test_files:
                    shutil.copy2(test_file, temp_dir)
                
                # Change to temp directory
                os.chdir(temp_dir)
                
                # Write solution files in temp directory
                self.file_manager.write_files(solution_files)
                
                # Run tests in temp directory
                test_results, _ = self.test_manager.run_and_parse()
                
            finally:
                # Always restore original directory
                os.chdir(original_cwd)
                
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass  # Ignore cleanup errors
            
            generation_time = time.time() - start_time
            
            candidate = SolutionCandidate(
                solution_files=solution_files,
                test_results=test_results,
                architecture_name=arch_name,
                generation_time=generation_time
            )
            
            # If this is a perfect solution, signal other threads to stop
            if candidate.is_perfect:
                with self.termination_lock:
                    self.perfect_solution_found = True
                print(f"[PARALLEL] ⚡ Perfect solution found: '{arch_name}' - signaling early termination")
            
            return candidate
        
        except Exception as e:
            print(f"[PARALLEL] Error generating solution: {e}")
            return SolutionCandidate(
                solution_files={},
                test_results=None,
                architecture_name=architecture_hint or "failed",
                generation_time=time.time() - start_time
            )
    
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
        
        candidates = []
        use_dynamic_queue = True  # Enable dynamic architecture generation
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit initial tasks
            future_to_hint = {
                executor.submit(self.generate_and_test_solution, problem, hint): hint
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
                        executor.shutdown(wait=False, cancel_futures=True)
                        
                        # Break out of the loop
                        break
                    elif self.perfect_solution_found:
                        # Perfect solution found by another thread but not yet in candidates list
                        # Continue processing to get it
                        print(f"[DEBUG] Perfect solution flag set but not in candidates yet, continuing to collect...")
                    
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
    Called by benchmark system or Docker sandbox with problem data.
    
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
    
    print(f"[AGENT] Test-Driven Agent initialized - Mode: {mode or 'AUTO'}")
    print(f"\n{'='*80}")
    print("TEST-DRIVEN ITERATIVE AGENT")
    print("="*80)
    print(f"[SETUP] Working directory: {os.getcwd()}")
    print(f"[SETUP] Problem type: {mode or 'AUTO-DETECT'}")
    
    # Detect if we're in Docker sandbox or local benchmark
    in_docker_sandbox = os.path.exists("/sandbox")
    print(f"[SETUP] Environment: {'Docker Sandbox' if in_docker_sandbox else 'Local Benchmark'}")
    
    # Change to repo directory
    # Docker sandbox: /sandbox/repo/
    # Local benchmark: workspace/repo/
    if in_docker_sandbox:
        repo_dir = "/sandbox/repo"
    else:
        repo_dir = os.path.join(os.getcwd(), "repo")
    
    if os.path.exists(repo_dir) and os.path.isdir(repo_dir):
        print(f"[SETUP] Changing to repo directory: {repo_dir}")
        os.chdir(repo_dir)
    else:
        print(f"[SETUP] Warning: Repo directory not found: {repo_dir}")
        print(f"[SETUP] Staying in current directory: {os.getcwd()}")
    
    # Initialize git repo and commit skeleton files
    # This is CRITICAL for generating proper patches
    try:
        # Check if git repo exists
        result = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, cwd=".")
        if result.returncode != 0:
            # Initialize git repo
            print("[SETUP] Initializing git repository...")
            subprocess.run(["git", "init"], capture_output=True, cwd=".", check=True)
            subprocess.run(["git", "config", "user.email", "agent@test.com"], capture_output=True, cwd=".")
            subprocess.run(["git", "config", "user.name", "Test Agent"], capture_output=True, cwd=".")
            
            # Commit skeleton files (all Python files in current directory)
            # Use shell=True for glob expansion
            add_result = subprocess.run("git add *.py", shell=True, capture_output=True, cwd=".", text=True)
            if add_result.returncode != 0:
                print(f"[SETUP] Warning: Git add failed: {add_result.stderr}")
            
            commit_result = subprocess.run(["git", "commit", "-m", "Initial skeleton"], capture_output=True, cwd=".", text=True)
            if commit_result.returncode == 0:
                print("[SETUP] ✓ Git repository initialized with skeleton files")
            else:
                # Check if there were no files to commit
                if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
                    print("[SETUP] ✓ Git repository initialized (no skeleton files to commit)")
                else:
                    print(f"[SETUP] Warning: Git commit failed: {commit_result.stderr}")
        else:
            print("[SETUP] Git repository already exists")
            # Even if git exists, commit any uncommitted skeleton files
            add_result = subprocess.run("git add *.py", shell=True, capture_output=True, cwd=".", text=True)
            commit_result = subprocess.run(["git", "commit", "-m", "Commit skeleton files"], capture_output=True, cwd=".", text=True)
            if commit_result.returncode == 0:
                print("[SETUP] ✓ Committed skeleton files")
    except Exception as e:
        print(f"[SETUP] Warning: Could not initialize git: {e}")
    
    # Auto-detect mode if not specified
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
    
    print(f"[AGENT] Test-Driven Agent initialized - Mode: {mode.upper()}")
    
    # Create concrete implementations
    # API URL: Use sandbox proxy in Docker, localhost in local benchmark
    api_url = f"{SANDBOX_PROXY_URL}/v1/chat/completions" if in_docker_sandbox else "http://localhost:8000/v1/chat/completions"
    print(f"[SETUP] Using API URL: {api_url}")
    
    test_runner = PytestRunner()
    code_generator = LLMCodeGenerator(api_url=api_url)
    arch_generator = LLMArchitectureGenerator(api_url=api_url)
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
        
        return patch
    
    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

# ============================================================================
# Main Entry Point - Docker Sandbox Compatible
# ============================================================================

if __name__ == "__main__":
    import json
    import traceback
    
    # Check if running in Docker sandbox (expects /sandbox/input.json and /sandbox/output.json)
    if os.path.exists("/sandbox/input.json"):
        print("[MAIN] Running in Docker sandbox mode")
        
        try:
            # Read input from sandbox
            with open("/sandbox/input.json", "r") as f:
                input_data = json.load(f)
            
            print(f"[MAIN] Loaded input: {list(input_data.keys())}")
            
            # Run agent
            patch = agent_main(input_data)
            
            # Write output
            output = {
                "success": True,
                "output": patch
            }
            
            with open("/sandbox/output.json", "w") as f:
                json.dump(output, f, indent=2)
            
            print("[MAIN] ✓ Completed successfully")
            
        except Exception as e:
            print(f"[MAIN] ✗ Agent failed: {e}")
            traceback.print_exc()
            
            # Write error output
            output = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output": ""
            }
            
            with open("/sandbox/output.json", "w") as f:
                json.dump(output, f, indent=2)
    
    else:
        # Local testing mode - run example
        print("[MAIN] Running in local testing mode")
        
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
