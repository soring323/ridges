"""
Test-Driven Iterative Agent
============================
A streamlined agent that uses test-driven development for superior accuracy and efficiency.

Key Innovations:
1. Single-pass solution generation (not 3x)
2. Single-pass test generation (not 15x)
3. Immediate test execution with feedback loop
4. Iterative refinement based on test failures
5. Execution-based validation (not comment analysis)
6. Smart model routing per operation
7. Semantic context compression

Strategy:
- CREATE: Generate → Test → Fix failures iteratively
- FIX: Search → Edit → Test → Refine iteratively

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
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

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
You are an expert Python unittest testcase developer. 
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
# Configuration
# ============================================================================

RUN_ID = os.getenv("RUN_ID", str(uuid4()))
SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))

# Model selection - use right model for right task
REASONING_MODEL = "deepseek-ai/DeepSeek-V3-0324"  # Complex reasoning
CODING_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"  # Code generation
FAST_MODEL = "deepseek-ai/DeepSeek-V3-0324"  # Quick operations

MAX_ITERATIONS = 10  # Max refinement iterations
MAX_FIX_STEPS = 100  # Max steps for FIX mode

print(f"[AGENT] Test-Driven Agent initialized - RUN_ID: {RUN_ID}")

# ============================================================================
# Network Layer
# ============================================================================

def call_llm(messages: List[Dict], model: str = CODING_MODEL, temperature: float = 0.0, max_retries: int = 3) -> str:
    """Call LLM with retry logic and error handling.
    
    Temperature strategy:
    - 0.0: Deterministic (tests, bug fixes, file selection)
    - 0.3: Balanced creativity (initial solution generation)
    - 0.8: High creativity (alternative architectures for diversity)
    """
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
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'choices' in result:
                    return result['choices'][0]['message']['content']
                elif isinstance(result, str):
                    return result.strip()
                else:
                    return str(result)
            else:
                print(f"[NETWORK] HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
                
        except requests.exceptions.Timeout:
            print(f"[NETWORK] Timeout (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"[NETWORK] Error: {e} (attempt {attempt + 1}/{max_retries})")
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts")

def are_architectures_similar(name1: str, name2: str) -> bool:
    """Check if two architecture names are similar based on common words.
    
    Args:
        name1: First architecture name
        name2: Second architecture name
    
    Returns:
        True if architectures share at least 2 significant words
    """
    # Word-based similarity check
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())
    common_words = words1 & words2
    
    # Need at least 2 significant words in common
    if len(common_words) >= 2:
        print(f"[SIMILARITY] '{name1}' vs '{name2}': {len(common_words)} common words")
        return True
    
    return False

# ============================================================================
# Utility Functions
# ============================================================================

def extract_python_code(text: str) -> str:
    """Extract Python code from markdown or raw text."""
    # Try to find code blocks
    pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no code blocks, return as-is (might be raw code)
    return text.strip()

def parse_file_blocks(text: str) -> Dict[str, str]:
    """Parse text containing multiple files in format: filename.py\n<code>\n\n"""
    files = {}
    
    # Try to extract from code blocks first
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        text = '\n\n'.join(code_blocks)
    
    lines = text.split('\n')
    current_file = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        # Check if line is a filename (more flexible matching)
        if stripped.endswith('.py') and len(stripped) < 100 and '(' not in stripped:
            # Save previous file
            if current_file and current_content:
                content = '\n'.join(current_content).strip()
                if content:  # Only save non-empty content
                    files[current_file] = content
            # Start new file
            current_file = stripped
            current_content = []
        elif current_file:
            current_content.append(line)
    
    # Save last file
    if current_file and current_content:
        content = '\n'.join(current_content).strip()
        if content:
            files[current_file] = content
    
    # If no files found, try to treat entire text as main.py
    if not files and text.strip():
        # Check if it looks like Python code
        if 'def ' in text or 'class ' in text or 'import ' in text:
            files['main.py'] = text.strip()
    
    return files

def write_files(files: Dict[str, str]) -> List[str]:
    """Write files to disk and return list of created files."""
    created = []
    for filename, content in files.items():
        try:
            # Create directories if needed
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                f.write(content)
            created.append(filename)
            print(f"[FILES] Created: {filename}")
        except Exception as e:
            print(f"[FILES] Error writing {filename}: {e}")
    return created

def run_tests(test_file: Optional[str] = None, timeout: int = 30) -> Tuple[bool, str]:
    """Run tests and return (success, output)."""
    try:
        # First, check what test files exist
        test_files_found = []
        for f in Path('.').glob('*.py'):
            if f.name.startswith('test_') or f.name == 'tests.py':
                test_files_found.append(str(f))
        
        if not test_files_found and not test_file:
            return False, "[TEST ERROR] No test files found in current directory"
        
        if test_file:
            cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
        else:
            # Explicitly specify test files if found
            if test_files_found:
                cmd = ["python", "-m", "pytest"] + test_files_found + ["-v", "--tb=short"]
            else:
                cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        output = result.stdout + "\n" + result.stderr
        success = result.returncode == 0
        
        # Add debug info if no tests collected
        if "collected 0 items" in output or "no tests ran" in output.lower():
            output += f"\n[DEBUG] Test files in directory: {test_files_found}"
            output += f"\n[DEBUG] Current directory: {os.getcwd()}"
            output += f"\n[DEBUG] All .py files: {list(Path('.').glob('*.py'))}"
        
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Test execution error: {e}"

def parse_test_results(output: str) -> Dict[str, Any]:
    """Parse pytest output to extract detailed test results."""
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "passed_tests": [],
        "failed_tests": [],
        "error_details": []
    }
    
    # Parse test counts
    passed_match = re.search(r'(\d+) passed', output)
    failed_match = re.search(r'(\d+) failed', output)
    error_match = re.search(r'(\d+) error', output)
    
    results["passed"] = int(passed_match.group(1)) if passed_match else 0
    results["failed"] = int(failed_match.group(1)) if failed_match else 0
    results["errors"] = int(error_match.group(1)) if error_match else 0
    results["total"] = results["passed"] + results["failed"] + results["errors"]
    
    # Parse individual test results
    for line in output.split('\n'):
        if ' PASSED' in line:
            test_name = line.split('::')[-1].split(' PASSED')[0].strip()
            results["passed_tests"].append(test_name)
        elif ' FAILED' in line:
            test_name = line.split('::')[-1].split(' FAILED')[0].strip()
            results["failed_tests"].append(test_name)
    
    # Extract error details from FAILURES section
    if '=== FAILURES ===' in output or '=== ERRORS ===' in output:
        failure_section = output.split('=== FAILURES ===')[-1] if '=== FAILURES ===' in output else ""
        if failure_section:
            # Extract first few lines of each failure
            current_test = None
            error_lines = []
            for line in failure_section.split('\n')[:100]:  # Limit to avoid huge logs
                if line.startswith('_'):
                    if current_test and error_lines:
                        results["error_details"].append({
                            "test": current_test,
                            "error": '\n'.join(error_lines[:5])  # First 5 lines
                        })
                    current_test = line.strip('_ ')
                    error_lines = []
                elif line.strip() and current_test:
                    error_lines.append(line)
            
            # Add last test
            if current_test and error_lines:
                results["error_details"].append({
                    "test": current_test,
                    "error": '\n'.join(error_lines[:5])
                })
    
    return results

def get_git_diff() -> str:
    """Get git diff of all changes."""
    try:
        # Stage all Python files
        subprocess.run(["git", "add", "*.py"], capture_output=True)
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except Exception as e:
        print(f"[GIT] Error getting diff: {e}")
        return ""

def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check if Python code has syntax errors."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

# ============================================================================
# Problem Type Detection
# ============================================================================

def detect_problem_type(problem_statement: str) -> str:
    """Detect if problem is CREATE or FIX type."""
    # Check for existing Python files
    py_files = list(Path('.').rglob('*.py'))
    has_existing_code = len(py_files) > 0
    
    # Simple heuristic
    keywords_fix = ['fix', 'bug', 'error', 'issue', 'broken', 'incorrect', 'modify', 'update']
    keywords_create = ['create', 'implement', 'write', 'build', 'generate']
    
    statement_lower = problem_statement.lower()
    
    fix_score = sum(1 for kw in keywords_fix if kw in statement_lower)
    create_score = sum(1 for kw in keywords_create if kw in statement_lower)
    
    if has_existing_code and fix_score > create_score:
        return "FIX"
    else:
        return "CREATE"

# ============================================================================
# CREATE Mode - Test-Driven Development
# ============================================================================

def post_workflow(problem_statement: str, solution_files: Dict[str, str], timeout: int, start_time: float, mode: str = "CREATE") -> str:
    """
    Unified post workflow for both CREATE and FIX modes.
    
    Strategy:
    1. Run tests iteratively
    2. Fix failures with fix_test_failures
    3. If stuck → try alternative architecture
    4. Repeat until success or timeout
    
    Args:
        problem_statement: The problem to solve
        solution_files: Current solution files
        timeout: Timeout in seconds
        start_time: Start time for timeout calculation
        mode: "CREATE" or "FIX" for logging purposes
    
    Returns:
        Git diff patch
    """
    print(f"\n[POST WORKFLOW] Starting test-driven refinement ({mode} mode)...")
    
    previous_failed = set()
    stuck_count = 0
    max_alternatives = 10  # Try up to 10 different architectures
    alternatives_tried = 0
    tried_architectures = {}  # Track: {architecture_name: test_score}
    
    for iteration in range(MAX_ITERATIONS):
        if time.time() - start_time > timeout - 60:
            print("[TIMEOUT] Stopping refinement")
            break
        
        print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")
        
        # Run tests
        success, output = run_tests()
        test_results = parse_test_results(output)
        
        # Detailed test logging
        print(f"[TESTS] {test_results['passed']}/{test_results['total']} passed, {test_results['failed']} failed")
        
        if test_results['passed_tests']:
            print(f"[TESTS] ✓ Passed: {', '.join(test_results['passed_tests'][:5])}" + 
                  (f" (+{len(test_results['passed_tests'])-5} more)" if len(test_results['passed_tests']) > 5 else ""))
        
        if test_results['failed_tests']:
            print(f"[TESTS] ✗ Failed: {', '.join(test_results['failed_tests'][:5])}" + 
                  (f" (+{len(test_results['failed_tests'])-5} more)" if len(test_results['failed_tests']) > 5 else ""))
        
        # Show first few error details
        if test_results['error_details']:
            print("[ERRORS] Sample failures:")
            for i, error in enumerate(test_results['error_details'][:3]):
                print(f"  {i+1}. {error['test']}")
                for line in error['error'].split('\n')[:2]:
                    if line.strip():
                        print(f"     {line.strip()}")
        
        if success:
            print("[SUCCESS] All tests passed!")
            break
        
        # Check if we're stuck (same failures for 2+ iterations = immediate detection)
        current_failed = set(test_results['failed_tests'])
        if current_failed == previous_failed and len(current_failed) > 0:
            stuck_count += 1
            # Detect stuck faster: 2 iterations instead of 3
            if stuck_count >= 2 and alternatives_tried < max_alternatives:
                print(f"\n[STUCK] Same {len(current_failed)} test(s) failing for {stuck_count} iterations")
                print(f"[STUCK] Trying alternative architecture #{alternatives_tried + 1}/{max_alternatives}...")
                
                # Save current best solution
                best_solution = solution_files.copy()
                best_score = test_results['passed']
                print(f"[BACKUP] Saved current solution: {best_score}/{test_results['total']} passed")
                
                # Try alternative architecture with attempt number for variety
                alternative = try_alternative_architecture(
                    problem_statement, 
                    output, 
                    solution_files,
                    test_results,
                    attempt=alternatives_tried + 1
                )
                
                if alternative:
                    # Extract architecture name
                    arch_name = alternative.pop('__architecture_name__', f"Architecture #{alternatives_tried + 1}")
                    
                    # Check if we've tried this architecture before (semantic similarity)
                    is_duplicate = False
                    for tried_name in tried_architectures.keys():
                        if are_architectures_similar(arch_name, tried_name):
                            is_duplicate = True
                            print(f"[ALTERNATIVE #{alternatives_tried + 1}] ⚠️ Similar to '{tried_name}' (already tried), skipping...")
                            break
                    
                    if is_duplicate:
                        # Don't count as attempt, stay in alternative-seeking mode
                        # Keep stuck count high to try next alternative
                        stuck_count = 2
                        continue
                    
                    # Now we're actually using this alternative, increment counter
                    alternatives_tried += 1
                    
                    print(f"[ALTERNATIVE #{alternatives_tried}] Architecture: {arch_name}")
                    print(f"[ALTERNATIVE #{alternatives_tried}] Generated new architecture, iterating to improve...")
                    solution_files.update(alternative)
                    write_files(alternative)
                    
                    # Give alternative architecture iterations to improve
                    # Always give at least 3 iterations, up to 4 max
                    alt_iterations = max(3, min(4, MAX_ITERATIONS - iteration - 1))
                    print(f"[ALTERNATIVE #{alternatives_tried}] Running {alt_iterations} refinement iterations...")
                    
                    alt_stuck = 0
                    alt_prev_failed = set()
                    
                    for alt_iter in range(alt_iterations):
                        alt_success, alt_output = run_tests()
                        alt_results = parse_test_results(alt_output)
                        print(f"[ALTERNATIVE #{alternatives_tried}] Iteration {alt_iter+1}/{alt_iterations}: {alt_results['passed']}/{alt_results['total']} passed")
                        
                        # Show failure details for alternative
                        if alt_results['failed_tests']:
                            print(f"[ALTERNATIVE #{alternatives_tried}] ✗ Failed: {', '.join(alt_results['failed_tests'][:3])}" + 
                                  (f" (+{len(alt_results['failed_tests'])-3} more)" if len(alt_results['failed_tests']) > 3 else ""))
                            # Show sample error details
                            if alt_results.get('error_details'):
                                print(f"[ALTERNATIVE #{alternatives_tried}] Sample error:")
                                error = alt_results['error_details'][0]
                                for line in error['error'].split('\n')[:3]:
                                    if line.strip():
                                        print(f"     {line.strip()}")
                        
                        if alt_success:
                            print(f"[ALTERNATIVE #{alternatives_tried}] ✓ All tests passed!")
                            break
                        
                        # Check if alternative is also stuck
                        alt_curr_failed = set(alt_results['failed_tests'])
                        if alt_curr_failed == alt_prev_failed:
                            alt_stuck += 1
                            if alt_stuck >= 2:
                                print(f"[ALTERNATIVE #{alternatives_tried}] Stuck on same {len(alt_curr_failed)} test(s), stopping early")
                                break
                        else:
                            alt_stuck = 0
                        alt_prev_failed = alt_curr_failed
                        
                        # Try to fix alternative architecture
                        print(f"[ALTERNATIVE #{alternatives_tried}] Analyzing failures and generating fix...")
                        alt_fixed = fix_test_failures(problem_statement, alt_output, solution_files)
                        if alt_fixed:
                            solution_files.update(alt_fixed)
                            write_files(alt_fixed)
                        else:
                            print(f"[ALTERNATIVE #{alternatives_tried}] Could not generate fix")
                            break
                    
                    # Final test of alternative
                    final_success, final_output = run_tests()
                    final_results = parse_test_results(final_output)
                    
                    # Record this architecture attempt
                    tried_architectures[arch_name] = final_results['passed']
                    
                    print(f"\n[COMPARISON] Original: {best_score}/{test_results['total']} | {arch_name}: {final_results['passed']}/{final_results['total']}")
                    
                    # Keep the better solution
                    if final_results['passed'] > best_score:
                        print(f"[ALTERNATIVE #{alternatives_tried}] ✓ Better! Continuing with '{arch_name}'...")
                        stuck_count = 0
                        previous_failed = set(final_results['failed_tests'])
                        continue
                    else:
                        print(f"[ALTERNATIVE #{alternatives_tried}] ✗ Not better, reverting...")
                        solution_files.update(best_solution)
                        write_files(best_solution)
                        stuck_count = 0  # Reset to try next alternative
                        previous_failed = current_failed
                        continue
                else:
                    print(f"[STUCK] Could not generate alternative #{alternatives_tried}")
                    # Reset stuck_count to try normal fixes before attempting another alternative
                    stuck_count = 0
                    
            elif stuck_count >= 2 and alternatives_tried >= max_alternatives:
                print(f"\n[STUCK] Tried {alternatives_tried} alternatives, all failed. Stopping.")
                break
        else:
            stuck_count = 0
        previous_failed = current_failed
        
        # Parse failures and fix
        print("[FIXING] Analyzing failures and generating fix...")
        fixed = fix_test_failures(problem_statement, output, solution_files)
        
        if not fixed:
            print("[WARNING] Could not generate fix")
            break
        
        # Update files
        solution_files.update(fixed)
        write_files(fixed)
    
    # Show summary of tried architectures
    if tried_architectures:
        print(f"\n[SUMMARY] Tried {len(tried_architectures)} alternative architecture(s):")
        for arch_name, score in tried_architectures.items():
            print(f"  - {arch_name}: {score} tests passed")
    
    # Return final patch
    patch = get_git_diff()
    print(f"\n[COMPLETE] Generated patch ({len(patch)} bytes)")
    return patch

def create_mode(problem_statement: str, timeout: int) -> str:
    """
    CREATE mode: Generate solution then run post workflow.
    
    Strategy:
    1. Generate initial solution
    2. Generate comprehensive tests
    3. Run post workflow (unified with FIX mode)
    """
    print("\n" + "="*80)
    print("CREATE MODE - Test-Driven Development")
    print("="*80)
    
    start_time = time.time()

    code_skeleton = get_code_skeleton()
    
    # Step 1: Generate initial solution
    print("\n[STEP 1] Generating initial solution...")
    solution_files = generate_solution(problem_statement)
    
    if not solution_files:
        print("[ERROR] Failed to generate solution")
        return ""
    
    created_files = write_files(solution_files)
    print(f"[STEP 1] Created {len(created_files)} solution files")
    
    # Step 2: Generate tests
    print("\n[STEP 2] Generating test suite...")
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    print(test_cases)
    # Extract and write files from test cases
    test_files = extract_and_write_files(test_cases)
    if not test_files:
        print("[ERROR] Failed to generate tests")
        return get_git_diff()
    print(f"Created or Updated {len(test_files)} files: {test_files}")
    
    # Run initial tests to see what we're working with
    print("[STEP 2] Running initial tests to establish baseline...")
    success, output = run_tests()
    initial_results = parse_test_results(output)
    print(f"[STEP 2] Baseline: {initial_results['passed']}/{initial_results['total']} tests passing")
    if initial_results['total'] > 0:
        print(f"[STEP 2] Test coverage: {len(initial_results['passed_tests'])} passing, {len(initial_results['failed_tests'])} failing")
    
    # Step 3: Run unified post workflow
    print("\n[STEP 3] Running post workflow...")
    return post_workflow(problem_statement, solution_files, timeout, start_time, mode="CREATE")

def generate_solution(problem_statement: str) -> Dict[str, str]:
    """Generate initial solution with better reasoning."""
    
    # Check if tests.py exists to show examples
    test_examples = ""
    if Path('tests.py').exists():
        with open('tests.py', 'r') as f:
            test_content = f.read()
            # Extract first few test methods as examples
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
    
    prompt = f"""You are an expert Python developer. Use step-by-step reasoning to solve this problem.

Problem Statement:
{problem_statement}
{test_examples}

STEP 1 - ANALYZE THE PROBLEM:
- What is the core problem asking for?
- What are the key requirements and constraints?
- What edge cases need to be handled?
- What data structures or algorithms are most appropriate?

STEP 2 - DESIGN THE SOLUTION:
- What classes/functions are needed?
- How should they interact?
- What is the overall architecture?
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
        print("[DEBUG] Calling LLM for solution generation (using reasoning model)...")
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,  # Use reasoning model for better architecture
            temperature=0.0  # Balanced creativity for initial solution
        )
        
        print(f"[DEBUG] Got response ({len(response)} chars)")
        print(f"[DEBUG] First 200 chars: {response[:200]}...")
        
        # Extract code
        code = extract_python_code(response)
        print(f"[DEBUG] Extracted code ({len(code)} chars)")
        
        files = parse_file_blocks(code if code else response)
        print(f"[DEBUG] Parsed {len(files)} files: {list(files.keys())}")
        
        if not files:
            print("[ERROR] No files parsed from response")
            print(f"[DEBUG] Full response:\n{response}")
            return {}
        
        # Validate syntax
        for filename, content in files.items():
            valid, error = check_syntax(content)
            if not valid:
                print(f"[SYNTAX] Error in {filename}: {error}")
            else:
                print(f"[SYNTAX] ✓ {filename} is valid")
        
        return files
    except Exception as e:
        print(f"[ERROR] Solution generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def generate_tests(problem_statement: str, solution_files: Dict[str, str]) -> Dict[str, str]:
    """Generate test suite - single pass, not 15x."""
    
    solution_summary = "\n\n".join([f"{name}:\n{content[:500]}..." for name, content in solution_files.items()])
    
    prompt = f"""You are an expert test developer. Generate comprehensive tests for this solution.

Problem Statement:
{problem_statement}

Solution Files:
{solution_summary}

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
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=CODING_MODEL
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        return files
    except Exception as e:
        print(f"[ERROR] Test generation failed: {e}")
        return {}

def analyze_edge_case_from_test(test_name: str, test_output: str) -> str:
    """Analyze the specific edge case from test name and output."""
    insights = []
    
    # Extract key insight from test name - GENERIC patterns
    # Look for common test naming patterns that reveal edge cases
    
    # Negative conditions (should NOT happen)
    if "should_not" in test_name or "shouldnt" in test_name or "not_be" in test_name:
        insights.append("**Edge Case**: Test name contains 'should_not' - something should NOT happen in this scenario")
    
    # "Only" conditions (exclusive behavior)
    if "only_" in test_name or "_only" in test_name:
        insights.append("**Requirement**: Test name contains 'only' - behavior should happen exclusively under certain conditions")
    
    # Change/update patterns
    if "change" in test_name and "but" in test_name:
        insights.append("**Edge Case**: Test name has 'change...but' pattern - some changes should not trigger certain effects")
    
    # Multiple/many patterns (handling multiple items)
    if "multiple" in test_name or "many" in test_name:
        insights.append("**Edge Case**: Test involves multiple items - check for proper handling of collections")
    
    # Empty/zero patterns
    if "empty" in test_name or "zero" in test_name or "no_" in test_name:
        insights.append("**Edge Case**: Test involves empty/zero case - boundary condition")
    
    # Already/duplicate patterns
    if "already" in test_name or "duplicate" in test_name or "twice" in test_name:
        insights.append("**Edge Case**: Test involves repeated operations - check for idempotency or duplicate handling")
    
    # Parse test name for semantic meaning (convert snake_case to readable)
    readable_name = test_name.replace('_', ' ').replace('test ', '')
    insights.append(f"**Test Intent**: '{readable_name}'")
    
    # Analyze test output for additional clues
    if test_output:
        # Check for assertion errors that reveal expected vs actual
        if "AssertionError:" in test_output:
            # Extract expected vs actual values
            if "!=" in test_output:
                # Pattern: actual != expected
                match = re.search(r'AssertionError:\s*(.+?)\s*!=\s*(.+?)(?:\n|$)', test_output)
                if match:
                    actual = match.group(1).strip()
                    expected = match.group(2).strip()
                    insights.append(f"**Assertion**: Expected `{expected}` but got `{actual}`")
            elif "==" in test_output:
                # Pattern: self.assertEqual(actual, expected)
                match = re.search(r'assertEqual\((.+?),\s*(.+?)\)', test_output)
                if match:
                    actual_var = match.group(1).strip()
                    expected_val = match.group(2).strip()
                    insights.append(f"**Assertion**: `{actual_var}` should equal `{expected_val}`")
        
        # Check for common error patterns
        if "[] !=" in test_output or "assertEqual(cb" in test_output:
            insights.append("**Pattern**: Callback observer list should be empty but contains values")
        
        if "AttributeError" in test_output:
            insights.append("**Error**: Missing attribute or method - implementation incomplete")
        
        if "TypeError" in test_output:
            insights.append("**Error**: Type mismatch - check function signatures and parameters")
    
    if insights:
        return "\n## Edge Case Analysis:\n" + "\n".join(f"- {i}" for i in insights) + "\n"
    return ""

def extract_problem_domain_hints(problem_statement: str, current_files: Dict[str, str]) -> str:
    """Use LLM to analyze problem domain and suggest relevant patterns (CoT approach)."""
    
    # Get a sample of current code
    code_sample = ""
    if current_files:
        # Take first 500 chars of each file
        samples = []
        for name, content in list(current_files.items())[:2]:
            samples.append(f"{name}:\n{content[:500]}")
        code_sample = "\n\n".join(samples)
    
    prompt = f"""Analyze this problem and current implementation to identify the domain and relevant patterns.

Problem Statement:
{problem_statement}

Current Implementation (sample):
{code_sample if code_sample else "No implementation yet"}

Provide a brief analysis (2-3 sentences max):
1. What domain is this? (e.g., reactive systems, API, data structures, algorithms, etc.)
2. What design patterns are relevant? (e.g., Observer, Strategy, Factory, etc.)
3. What is the key technical challenge based on the problem description?

Be concise and specific. Focus on actionable insights."""

    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=FAST_MODEL,  # Use fast model for quick analysis
            temperature=0.3  # Low temperature for focused analysis
        )
        
        if response and len(response) > 20:
            return f"\n## Domain Analysis (LLM):\n{response.strip()}\n"
        
    except Exception as e:
        print(f"[DEBUG] Domain analysis failed: {e}")
    
    return ""

def fix_test_failures(problem_statement: str, test_output: str, current_files: Dict[str, str]) -> Dict[str, str]:
    """Fix code based on test failures."""
    
    # Extract failure information
    failure_summary = extract_failure_summary(test_output)
    
    # Extract failing test names for edge case analysis
    failed_tests = []
    for line in test_output.split('\n'):
        if 'FAILED' in line and '::' in line:
            test_name = line.split('::')[-1].split(' ')[0].strip()
            failed_tests.append(test_name)
    
    # Analyze edge cases from test names
    edge_case_analysis = ""
    if failed_tests:
        edge_case_analysis = analyze_edge_case_from_test(failed_tests[0], test_output)
        # Also get test code
        test_code = analyze_failing_test_code(failed_tests[0])
        edge_case_analysis += test_code
    
    current_code = "\n\n".join([f"{name}:\n{content}" for name, content in current_files.items()])
    
    prompt = f"""You are an expert Python debugger. Use systematic reasoning to fix the failing tests.

Problem Statement:
{problem_statement}

Current Implementation:
{current_code}

Test Failures (detailed):
{failure_summary}
{edge_case_analysis}

DEBUGGING PROCESS:

STEP 1 - UNDERSTAND THE FAILURES:
- What exactly is each test expecting?
- What is the actual vs expected behavior?
- Are there patterns in the failures (same type of error across multiple tests)?

STEP 2 - IDENTIFY ROOT CAUSES:
- What is the underlying bug causing these failures?
- Is it a logic error, missing functionality, or incorrect implementation?
- Are there edge cases not being handled?

STEP 3 - DESIGN THE FIX:
- What needs to change to fix the root cause?
- Will this fix break any currently passing tests?
- Is there a minimal change that addresses all related failures?

STEP 4 - IMPLEMENT THE FIX:
- Apply the fix to the code
- Ensure all edge cases are handled
- Maintain code quality and readability

Output the COMPLETE fixed code for each file in this format:
filename.py
```python
<complete fixed code>
```

Generate the fixed code now:"""

    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,
            temperature=0.0  # Precise fixes, no randomness
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        return files
    except Exception as e:
        print(f"[ERROR] Fix generation failed: {e}")
        return {}

def analyze_failing_test_code(test_name: str) -> str:
    """Read and extract the specific failing test code to understand edge case."""
    try:
        # Find test file
        test_files = list(Path('.').glob('*test*.py'))
        if not test_files:
            return ""
        
        # Read test file
        test_content = test_files[0].read_text()
        
        # Extract the specific test function
        # Look for: def test_name(...):
        pattern = rf'def {test_name}\(.*?\):(.*?)(?=\n    def |\n\nclass |\Z)'
        match = re.search(pattern, test_content, re.DOTALL)
        
        if match:
            test_code = match.group(0)
            return f"\n## Failing Test Code:\n```python\n{test_code}\n```\n"
        
        return ""
    except Exception as e:
        print(f"[DEBUG] Could not extract test code: {e}")
        return ""

def try_alternative_architecture(
    problem_statement: str,
    test_output: str, 
    current_files: Dict[str, str],
    test_results: Dict[str, Any],
    attempt: int = 1
) -> Dict[str, str]:
    """Generate alternative solution with different architectural approach using CoT reasoning."""
    
    failure_summary = extract_failure_summary(test_output)
    current_code = "\n\n".join([f"{name}:\n{content}" for name, content in current_files.items()])
    
    # Extract failing test code for better analysis
    test_code_analysis = ""
    failing_tests = test_results.get('failed_tests', [])
    if failing_tests:
        # Analyze the first failing test
        test_code_analysis = analyze_failing_test_code(failing_tests[0])
    
    # Extract domain hints from problem statement and current implementation
    domain_hints = extract_problem_domain_hints(problem_statement, current_files)
    
    # Identify what's failing
    failing_tests = test_results.get('failed_tests', [])
    
    # Use CoT to generate architecture-specific patterns
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

CRITICAL: Distinguish between bugs vs architectural limitations:
- **Bug**: Logic error, missing check, off-by-one → Can be fixed with small patch
- **Architectural Limitation**: Edge case reveals fundamental design flaw → Needs redesign

Ask yourself:
- Is this failing because of a simple bug, or because the architecture cannot handle this pattern?
- Does fixing this edge case require changing core data structures or control flow?
- Would fixing this break other parts or require extensive refactoring?
- Is the current approach fighting against the problem's natural structure?

If the edge case reveals an architectural limitation, a complete redesign is needed.

STEP 3 - GENERATE ALTERNATIVE ARCHITECTURES:
Propose 3 COMPLETELY DIFFERENT architectural approaches that could solve this problem.
For each alternative, explain:
- Architecture name and core concept
- Key design principles
- Why this approach addresses the root cause
- Trade-offs and considerations

**CRITICAL INSIGHT FROM FAILING TEST:**
The test name and error reveal the EXACT edge case to solve. Focus your architecture on THIS SPECIFIC REQUIREMENT, not general improvements.

For example:
- If test says "should_not_be_called_if_value_doesn't_change" → Architecture MUST compare old vs new values before triggering
- If test says "only_fire_once" → Architecture MUST deduplicate or batch events
- If test says "empty_list" → Architecture MUST handle empty/null cases

This is attempt #{attempt}, so focus on approaches that are FUNDAMENTALLY DIFFERENT from typical solutions.

STEP 4 - SELECT AND IMPLEMENT:
Choose the most promising approach (approach #{min(attempt, 3)}) from your alternatives.

**SELECTION CRITERIA (prioritize in this order):**
1. **Directly addresses the failing test's edge case** (highest priority)
   - Look at test name: what behavior is it checking?
   - Look at assertion: what value is expected vs actual?
   - Design architecture to satisfy THIS SPECIFIC requirement
   
2. **Simple to implement** (fewer moving parts = fewer bugs)
   - Prefer adding a single check/comparison over redesigning everything
   - Prefer modifying one method over changing multiple classes
   
3. **Minimal changes to working code** (preserve what already passes)
   - {test_results['passed']}/{test_results['total']} tests already pass
   - Don't break what works - only fix what's broken

**GENERAL ARCHITECTURAL PRINCIPLES:**
- If test checks "should NOT happen when X" → Add conditional check before action
- If test checks "only when Y changes" → Compare old vs new state
- If test checks "once per Z" → Add deduplication or batching
- If test checks "empty/null case" → Add boundary condition handling
- If multiple dependencies involved → Consider separating concerns or phases

**COMPLEXITY WARNING:**
- Simpler architectures succeed more often
- Complex redesigns often break passing tests
- If stuck after 3 attempts, try the SIMPLEST possible fix

Provide a complete implementation with:
- Clear explanation of the architecture
- How it solves the specific failing test cases
- Complete, working code

Problem Statement:
{problem_statement}

Current Implementation (STUCK - Attempt #{attempt}):
{current_code}

Persistent Test Failures:
{failure_summary}

Failing tests: {', '.join(failing_tests[:5])}
{test_code_analysis}

NOW, COMPLETE THE ANALYSIS AND IMPLEMENTATION:

Follow the 4-step process above to:
1. Identify the problem domain
2. Analyze the failure pattern
3. Generate 3 architectural alternatives
4. Select and implement approach #{min(attempt, 3)}

CRITICAL REQUIREMENTS:
- The current architecture CANNOT fix these failures through incremental changes
- You MUST redesign with a FUNDAMENTALLY DIFFERENT approach
- Do NOT try to patch the old code - start FRESH with new design
- Make sure the new architecture handles the edge case that's failing
- Provide COMPLETE working code, not just snippets

OUTPUT FORMAT:
First, show your reasoning:

## Domain Analysis:
[What type of problem is this and what patterns are relevant?]

## Root Cause Analysis:
[Why is the current approach failing?]

## Bug vs Architecture Decision:
Is this a **BUG** or **ARCHITECTURAL LIMITATION**?
- If BUG: [Explain why it's just a logic error]
- If ARCHITECTURAL: [Explain why the edge case reveals a fundamental design flaw]

## Edge Case Impact:
[How does this edge case expose limitations in the current design?]
[What would it take to fix this in the current architecture?]
[Why is a redesign the better approach?]

## Architectural Alternatives:
1. **[Architecture 1 name]**: [Core concept and why it handles the edge case]
2. **[Architecture 2 name]**: [Core concept and why it handles the edge case]
3. **[Architecture 3 name]**: [Core concept and why it handles the edge case]

## Selected Approach #{min(attempt, 3)}: [name]
[Detailed explanation of why this approach naturally handles the edge case]

Then provide the complete implementation:
filename.py
```python
<complete new implementation>
```

Generate the alternative architecture now:"""

    try:
        print("[ALTERNATIVE] Requesting architectural redesign from LLM...")
        # Lower temperature for more deterministic, focused solutions
        # Higher attempts = lower temperature (more focused on proven patterns)
        temp = max(0.3, 0.8 - (attempt * 0.15))  # 0.8 → 0.65 → 0.5 → 0.35 → 0.3
        print(f"[ALTERNATIVE] Using temperature {temp:.2f} (attempt #{attempt})")
        
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,
            temperature=temp
        )
        
        # Extract architecture name from response
        architecture_name = "Unknown Architecture"
        if "## Selected Approach" in response:
            # Extract the line with "## Selected Approach #X: Name"
            for line in response.split('\n'):
                if "## Selected Approach" in line and ":" in line:
                    architecture_name = line.split(':', 1)[1].strip()
                    break
        
        print(f"[ALTERNATIVE] Using: {architecture_name}")
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        if files:
            print(f"[ALTERNATIVE] Generated {len(files)} file(s) with new architecture")
            # Store architecture name in files dict for tracking
            files['__architecture_name__'] = architecture_name
        
        return files
    except Exception as e:
        print(f"[ERROR] Alternative generation failed: {e}")
        return {}

def extract_failure_summary(test_output: str, max_lines: int = 150) -> str:
    """Extract relevant failure information from test output with better structure."""
    lines = test_output.split('\n')
    
    # Extract the FAILURES section which has the most detail
    if '=== FAILURES ===' in test_output:
        failures_start = test_output.index('=== FAILURES ===')
        failures_section = test_output[failures_start:]
        
        # Also get the short summary at the end
        if '=== short test summary' in failures_section:
            summary_start = failures_section.index('=== short test summary')
            failures_section = failures_section[:summary_start]
        
        # Limit to reasonable size but include more context
        failure_lines = failures_section.split('\n')[:max_lines]
        result = '\n'.join(failure_lines)
        
        # Try to add the actual test code for context
        # Look for test file references and extract test code
        test_code_context = extract_test_code_context(test_output)
        if test_code_context:
            result += f"\n\n=== RELEVANT TEST CODE ===\n{test_code_context}"
        
        return result
    
    # Fallback: extract failed test names and errors
    failure_lines = []
    in_failure = False
    current_failure = []
    
    for line in lines:
        if line.startswith('_') and 'Test' in line:
            if current_failure:
                failure_lines.extend(current_failure[:10])  # First 10 lines of each failure
                failure_lines.append('')
            current_failure = [line]
            in_failure = True
        elif in_failure:
            current_failure.append(line)
            if line.startswith('=') or line.startswith('_'):
                in_failure = False
    
    if current_failure:
        failure_lines.extend(current_failure[:10])
    
    return '\n'.join(failure_lines) if failure_lines else test_output[-1000:]

def extract_test_code_context(test_output: str) -> str:
    """Extract actual test code from test files for failed tests."""
    import re
    
    # Find test file references like "tests.py:258"
    test_refs = re.findall(r'(tests?\.py):(\d+)', test_output)
    if not test_refs:
        return ""
    
    # Get unique test files
    test_files = set(ref[0] for ref in test_refs)
    
    context_lines = []
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                with open(test_file, 'r') as f:
                    test_content = f.read()
                
                # Extract the failing test methods
                for filename, line_num in test_refs[:3]:  # Limit to first 3 failures
                    if filename == test_file:
                        # Find the test method containing this line
                        lines = test_content.split('\n')
                        line_idx = int(line_num) - 1
                        
                        # Find start of test method (search backwards for "def test_")
                        start_idx = line_idx
                        while start_idx > 0:
                            if lines[start_idx].strip().startswith('def test_'):
                                break
                            start_idx -= 1
                        
                        # Find end of test method (next def or class)
                        end_idx = line_idx + 1
                        while end_idx < len(lines):
                            if lines[end_idx].strip().startswith('def ') or lines[end_idx].strip().startswith('class '):
                                break
                            end_idx += 1
                        
                        # Extract the test method
                        test_method = '\n'.join(lines[start_idx:min(end_idx, start_idx + 25)])  # Max 25 lines
                        context_lines.append(f"\nTest method from {filename}:{line_num}:\n{test_method}")
                        
            except Exception as e:
                continue
    
    return '\n'.join(context_lines[:500])  # Limit total context

# ============================================================================
# FIX Mode - Iterative Debugging
# ============================================================================

def fix_mode(problem_statement: str, timeout: int) -> str:
    """
    FIX mode: Assume implementation exists, then run post workflow.
    
    Strategy:
    1. Find relevant existing code
    2. Generate/find tests if needed
    3. Run post workflow (unified with CREATE mode)
    """
    print("\n" + "="*80)
    print("FIX MODE - Iterative Debugging")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Analyze problem and find relevant files (implementation already exists)
    print("\n[STEP 1] Finding relevant code...")
    relevant_files = find_relevant_files(problem_statement)
    print(f"[STEP 1] Found {len(relevant_files)} relevant files")
    
    if not relevant_files:
        print("[ERROR] No relevant files found")
        return ""
    
    # Step 2: Generate reproduction test if needed
    print("\n[STEP 2] Setting up tests...")
    test_file = generate_reproduction_test(problem_statement, relevant_files)
    
    if test_file:
        write_files(test_file)
        print("[STEP 2] Created reproduction test")
    else:
        print("[STEP 2] Using existing tests")
    
    # Step 3: Run unified post workflow
    print("\n[STEP 3] Running post workflow...")
    return post_workflow(problem_statement, relevant_files, timeout, start_time, mode="FIX")

def find_relevant_files(problem_statement: str) -> Dict[str, str]:
    """Find files relevant to the problem."""
    # Get all Python files
    py_files = list(Path('.').rglob('*.py'))
    
    # Read file contents
    files = {}
    for path in py_files:
        if 'test_' not in path.name:  # Skip test files
            try:
                with open(path, 'r') as f:
                    files[str(path)] = f.read()
            except Exception as e:
                print(f"[WARNING] Could not read {path}: {e}")
    
    # Use LLM to identify most relevant files
    if len(files) > 5:
        files = rank_relevant_files(problem_statement, files)
    
    return files

def rank_relevant_files(problem_statement: str, files: Dict[str, str]) -> Dict[str, str]:
    """Use LLM to rank and filter most relevant files."""
    file_summaries = "\n".join([f"{name}: {len(content)} chars" for name, content in files.items()])
    
    prompt = f"""Given this problem, identify the 3-5 most relevant files to modify.

Problem:
{problem_statement}

Available files:
{file_summaries}

Output only the filenames, one per line:"""

    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=FAST_MODEL,
            temperature=0.0  # Deterministic file selection
        )
        
        relevant_names = [line.strip() for line in response.split('\n') if line.strip()]
        return {name: content for name, content in files.items() if any(rn in name for rn in relevant_names)}
    except:
        # Fallback: return first 3 files
        return dict(list(files.items())[:3])

def generate_reproduction_test(problem_statement: str, relevant_files: Dict[str, str]) -> Dict[str, str]:
    """Generate a focused test to reproduce the issue."""
    
    code_summary = "\n\n".join([f"{name}:\n{content[:300]}..." for name, content in relevant_files.items()])
    
    prompt = f"""Generate a focused test that reproduces this issue.

Problem:
{problem_statement}

Relevant Code:
{code_summary}

Requirements:
1. Use pytest
2. Test should FAIL initially (reproduces the bug)
3. Keep it minimal and focused
4. Output format: test_bug.py followed by code

Generate the test:"""

    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=CODING_MODEL,
            temperature=0.0  # Deterministic test generation
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        return files
    except Exception as e:
        print(f"[ERROR] Test generation failed: {e}")
        return {}

# ============================================================================
# Main Entry Point
# ============================================================================

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo") -> str:
    """
    Main entry point for the agent.
    
    Args:
        input_dict: Dictionary with 'problem_statement' key
        repo_dir: Repository directory path
    
    Returns:
        Git diff patch as string
    """
    print("\n" + "="*80)
    print("TEST-DRIVEN ITERATIVE AGENT")
    print("="*80)
    
    # Setup
    problem_statement = input_dict.get("problem_statement", "")
    if not problem_statement:
        raise ValueError("No problem_statement provided")
    
    # Change to repo directory
    repo_path = os.path.abspath(repo_dir)
    if os.path.exists(repo_path):
        os.chdir(repo_path)
        print(f"[SETUP] Working directory: {repo_path}")
    
    # Initialize git if needed
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "agent@test.com"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Agent"], capture_output=True)
        subprocess.run(["git", "add", "-A"], capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], capture_output=True)
        print("[SETUP] Initialized git repository")
    
    # Detect problem type
    problem_type = detect_problem_type(problem_statement)
    print(f"[SETUP] Problem type: {problem_type}")
    
    # Calculate timeout
    timeout = DEFAULT_TIMEOUT - 120  # Reserve time for overhead
    
    try:
        # Route to appropriate mode
        if problem_type == "CREATE":
            patch = create_mode(problem_statement, timeout)
        else:
            patch = fix_mode(problem_statement, timeout)
        
        # Reset to clean state
        subprocess.run(["git", "reset", "--hard"], capture_output=True)
        
        return patch
    
    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        traceback.print_exc()
        
        # Try to return whatever changes we have
        try:
            patch = get_git_diff()
            subprocess.run(["git", "reset", "--hard"], capture_output=True)
            return patch
        except:
            return ""

if __name__ == "__main__":
    # Test the agent
    test_input = {
        "problem_statement": "Create a function that reverses a string"
    }
    result = agent_main(test_input)
    print(f"\nResult:\n{result}")

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = call_llm(test_generation_messages, model=CODING_MODEL)
            logger.info("Step 1 - Testcase Generation completed")
            
            # Step 5: Infinite Loop Check and Validation
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                }   
            ]
            
            testcode_checked_response = call_llm(testcases_check_messages, model=CODING_MODEL)
            logger.info("Step 2 - Testcase check completed")

            # Clean up the final response (use loop check response as it's the final validated version)
            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            
            lines = testcases.split("\n")
            if not lines[0].endswith(".py"):
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning testcase generation failed")
        return ""
    
    return ""

def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting test cases generation")
            
            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)
            
            if testcases:
                logger.info("Generated testcases successfully using multi-step reasoning")
                return testcases
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
                    }
                ]
                
                response = call_llm(messages, model=CODING_MODEL)
                
                # Clean up the response
                testcases = response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                
                logger.info("Generated testcases successfully using fallback approach")
                return testcases
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    import re
    
    created_files = []
    
    if not initial_solution.strip():
        print("No solution content to process")
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()
        
        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):  # Not a comment
            
            # Write the previous file if we have one
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                # Create directory if needed (for subdirectories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Join content and remove empty lines at start/end
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            
            # Start new file
            current_filename = stripped_line
            current_content = []
        else:
            # This line is content for the current file
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    
    return created_files

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