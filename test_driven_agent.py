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
import json
import subprocess
import requests
import time
import re
import ast
import traceback
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from pathlib import Path

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
    """Call LLM with retry logic and error handling."""
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

def create_mode(problem_statement: str, timeout: int) -> str:
    """
    CREATE mode: Generate solution with test-driven iterative refinement.
    
    Strategy:
    1. Generate initial solution (1x, not 3x)
    2. Generate comprehensive tests (1x, not 15x)
    3. Run tests → get failures
    4. Fix failures iteratively with test feedback
    5. Repeat until tests pass or max iterations
    """
    print("\n" + "="*80)
    print("CREATE MODE - Test-Driven Development")
    print("="*80)
    
    start_time = time.time()
    
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
    test_files = generate_tests(problem_statement, solution_files)
    
    if not test_files:
        print("[ERROR] Failed to generate tests")
        return get_git_diff()
    
    test_file_paths = write_files(test_files)
    print(f"[STEP 2] Created {len(test_file_paths)} test files")
    
    # Run initial tests to see what we're working with
    print("[STEP 2] Running initial tests to establish baseline...")
    success, output = run_tests()
    initial_results = parse_test_results(output)
    print(f"[STEP 2] Baseline: {initial_results['passed']}/{initial_results['total']} tests passing")
    if initial_results['total'] > 0:
        print(f"[STEP 2] Test coverage: {len(initial_results['passed_tests'])} passing, {len(initial_results['failed_tests'])} failing")
    
    # Step 3: Iterative refinement
    print("\n[STEP 3] Test-driven refinement...")
    
    previous_failed = set()
    stuck_count = 0
    max_alternatives = 10  # Try up to 5 different architectures
    alternatives_tried = 0
    
    for iteration in range(MAX_ITERATIONS):
        if time.time() - start_time > timeout - 60:
            print(f"[TIMEOUT] Stopping refinement")
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
            print(f"[ERRORS] Sample failures:")
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
                alternatives_tried += 1
                print(f"\n[STUCK] Same {len(current_failed)} test(s) failing for {stuck_count} iterations")
                print(f"[STUCK] Trying alternative architecture #{alternatives_tried}/{max_alternatives}...")
                
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
                    attempt=alternatives_tried
                )
                
                if alternative:
                    print(f"[ALTERNATIVE #{alternatives_tried}] Generated new architecture, iterating to improve...")
                    solution_files.update(alternative)
                    write_files(alternative)
                    
                    # Give alternative architecture iterations to improve
                    alt_iterations = min(4, MAX_ITERATIONS - iteration - 1)  # Use remaining iterations
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
                    
                    print(f"\n[COMPARISON] Original: {best_score}/{test_results['total']} | Alternative #{alternatives_tried}: {final_results['passed']}/{final_results['total']}")
                    
                    # Keep the better solution
                    if final_results['passed'] > best_score:
                        print(f"[ALTERNATIVE #{alternatives_tried}] ✓ Better! Continuing with new architecture...")
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
    
    # Return final patch
    patch = get_git_diff()
    print(f"\n[COMPLETE] Generated patch ({len(patch)} bytes)")
    return patch

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
            model=REASONING_MODEL  # Use reasoning model for better architecture
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

def fix_test_failures(problem_statement: str, test_output: str, current_files: Dict[str, str]) -> Dict[str, str]:
    """Fix code based on test failures."""
    
    # Extract failure information
    failure_summary = extract_failure_summary(test_output)
    
    current_code = "\n\n".join([f"{name}:\n{content}" for name, content in current_files.items()])
    
    prompt = f"""You are an expert Python debugger. Use systematic reasoning to fix the failing tests.

Problem Statement:
{problem_statement}

Current Implementation:
{current_code}

Test Failures (detailed):
{failure_summary}

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
            model=REASONING_MODEL
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        return files
    except Exception as e:
        print(f"[ERROR] Fix generation failed: {e}")
        return {}

def try_alternative_architecture(
    problem_statement: str,
    test_output: str, 
    current_files: Dict[str, str],
    test_results: Dict[str, Any],
    attempt: int = 1
) -> Dict[str, str]:
    """Generate alternative solution with different architectural approach."""
    
    failure_summary = extract_failure_summary(test_output)
    current_code = "\n\n".join([f"{name}:\n{content}" for name, content in current_files.items()])
    
    # Identify what's failing
    failing_tests = test_results.get('failed_tests', [])
    
    # Different architectural hints for each attempt
    architecture_hints = {
        1: """**PRIMARY APPROACH - Batched/Transactional Updates**:
- Collect all dependency changes in a batch
- Apply all updates atomically in one pass
- Only fire callbacks after ALL updates complete
- Track old values to detect actual changes""",
        
        2: """**SECONDARY APPROACH - Topological Sorting**:
- Build dependency graph of all cells
- Sort cells by topological order (dependencies first)
- Update cells in sorted order to avoid cascading issues
- Use depth-first search to determine update order""",
        
        3: """**TERTIARY APPROACH - Event Queue with Deduplication**:
- Use event queue instead of immediate propagation
- Deduplicate events for same cell
- Process queue in order, checking for value changes
- Only fire callbacks if final value differs from initial""",
        
        4: """**QUATERNARY APPROACH - Two-Phase Commit**:
- Phase 1: Compute all new values without updating state
- Phase 2: Update all values and fire callbacks only for actual changes
- Store old values before any updates
- Compare old vs new to determine which callbacks to fire""",
        
        5: """**QUINARY APPROACH - Lazy Evaluation with Dirty Flags**:
- Mark cells as dirty instead of immediate recomputation
- Only recompute when value is accessed
- Track which cells actually changed value
- Fire callbacks only after confirming value change"""
    }
    
    # Get hint for this attempt, or generate a unique variation
    if attempt in architecture_hints:
        hint = architecture_hints[attempt]
    else:
        # Generate variation by combining approaches
        hint = f"""**HYBRID APPROACH #{attempt}**:
- Combine elements from previous approaches
- Use creative variation: lazy evaluation + batching, or event queue + topological sort
- Focus on solving the cascading update problem with a novel combination
- Ensure callbacks only fire when final computed value actually changes"""
    
    prompt = f"""You are an expert Python architect. The current solution is stuck on edge cases that require a DIFFERENT ARCHITECTURAL APPROACH.

Problem Statement:
{problem_statement}

Current Implementation (STUCK - Attempt #{attempt}):
{current_code}

Persistent Test Failures:
{failure_summary}

Failing tests: {', '.join(failing_tests[:5])}

CRITICAL ANALYSIS:
The current architecture cannot fix these failures through incremental changes. You need to redesign with a FUNDAMENTALLY DIFFERENT approach.

RECOMMENDED ARCHITECTURE FOR ATTEMPT #{attempt}:
{hint}

Alternative patterns to consider:
1. **Batched/Transactional Updates**: Batch all changes and apply atomically
2. **Topological Sorting**: Sort updates by dependency graph
3. **Event Queue**: Use event queue instead of immediate propagation
4. **Immutable State**: Use immutable data structures
5. **Two-Phase Update**: Compute all new values first, then update and notify

INSTRUCTIONS:
1. Analyze WHY the current architecture fails (not just what fails)
2. Implement the RECOMMENDED ARCHITECTURE for this attempt
3. Write a COMPLETE REWRITE using the new pattern
4. Do NOT try to patch the old code - start fresh with new design
5. Make sure the new architecture handles the edge case that's failing

Output the COMPLETE alternative solution:
filename.py
```python
<complete new implementation>
```

Generate the alternative architecture now:"""

    try:
        print(f"[ALTERNATIVE] Requesting architectural redesign from LLM...")
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        if files:
            print(f"[ALTERNATIVE] Generated {len(files)} file(s) with new architecture")
        
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
    FIX mode: Iterative debugging with test feedback.
    
    Strategy:
    1. Understand the problem and locate relevant code
    2. Generate focused test to reproduce issue
    3. Apply fix
    4. Run tests to verify
    5. Repeat until fixed
    """
    print("\n" + "="*80)
    print("FIX MODE - Iterative Debugging")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Analyze problem and find relevant files
    print("\n[STEP 1] Analyzing problem...")
    relevant_files = find_relevant_files(problem_statement)
    print(f"[STEP 1] Found {len(relevant_files)} relevant files")
    
    # Step 2: Generate reproduction test
    print("\n[STEP 2] Generating reproduction test...")
    test_file = generate_reproduction_test(problem_statement, relevant_files)
    
    if test_file:
        write_files(test_file)
        print("[STEP 2] Created reproduction test")
    
    # Step 3: Iterative fixing
    print("\n[STEP 3] Iterative fixing...")
    
    for iteration in range(MAX_ITERATIONS):
        if time.time() - start_time > timeout - 60:
            print(f"[TIMEOUT] Stopping fixes")
            break
        
        print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")
        
        # Run tests
        success, output = run_tests()
        print(f"[TESTS] {'✓ PASSED' if success else '✗ FAILED'}")
        
        if success:
            print("[SUCCESS] All tests passed!")
            break
        
        # Apply fix
        print("[FIXING] Applying fix...")
        fixed = apply_fix(problem_statement, output, relevant_files)
        
        if not fixed:
            print("[WARNING] Could not generate fix")
            break
        
        # Update files
        write_files(fixed)
        relevant_files.update(fixed)
    
    # Return final patch
    patch = get_git_diff()
    print(f"\n[COMPLETE] Generated patch ({len(patch)} bytes)")
    return patch

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
            model=FAST_MODEL
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
            model=CODING_MODEL
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        return files
    except Exception as e:
        print(f"[ERROR] Test generation failed: {e}")
        return {}

def apply_fix(problem_statement: str, test_output: str, current_files: Dict[str, str]) -> Dict[str, str]:
    """Apply fix based on test failures."""
    
    failure_summary = extract_failure_summary(test_output)
    current_code = "\n\n".join([f"{name}:\n{content}" for name, content in current_files.items()])
    
    prompt = f"""Fix the code to resolve this issue.

Problem:
{problem_statement}

Current Code:
{current_code}

Test Failures:
{failure_summary}

Requirements:
1. Identify root cause
2. Apply minimal fix
3. Maintain backward compatibility
4. Output format: filename.py followed by fixed code

Generate the fix:"""

    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL
        )
        
        code = extract_python_code(response)
        files = parse_file_blocks(code if code else response)
        
        return files
    except Exception as e:
        print(f"[ERROR] Fix generation failed: {e}")
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
