"""
Competitive Programming Agent
==============================
A clean, focused agent for solving algorithm problems through structured reasoning.

Design Philosophy:
- Simple & Clear: ~400 lines vs complex multi-agent frameworks (3000+ lines)
- Single Purpose: Optimized for competitive programming, not general SWE tasks
- Four-Phase Approach: Analysis → Design → Implementation → Self-Testing
- Minimal Dependencies: Only Python standard library + basic infrastructure
- Self-Validating: Tests and debugs its own code before returning

Key Features:
- 🧠 Multi-phase reasoning: Deep problem analysis before coding
- 🔧 Self-testing: Runs unit tests and auto-debugs (up to 3 iterations)
- 📦 Standard library only: No external dependencies (numpy, pandas, etc.)
- 🔄 Retry logic: Robust LLM calls with exponential backoff
- 📝 Code extraction: Handles various LLM response formats (```python, raw, etc.)
- ✅ Import validation: Warns if forbidden packages detected

Inspired by complex agentic frameworks but intentionally kept simple and focused.
"""

import os
import subprocess
import requests
from uuid import uuid4


# Configuration
RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    RUN_ID = str(uuid4())  # Generate UUID if not provided
    os.environ["RUN_ID"] = RUN_ID  # Set for consistency

SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://localhost:8000")
MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

print(f"[AGENT] Initialized with RUN_ID: {RUN_ID}")


def call_llm(messages: list, max_retries: int = 3) -> str:
    """Call LLM via sandbox proxy with retry logic."""
    import json
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{SANDBOX_PROXY_URL}/api/inference",
                json={
                    "run_id": RUN_ID,
                    "model": MODEL,
                    "temperature": 0.0,
                    "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages]
                },
                timeout=120
            )
            
            if response.status_code == 200:
                # Properly decode JSON response
                result = response.text
                
                # Try to parse as JSON first (API might return JSON-encoded string)
                try:
                    # If response is a JSON string like "\"code here\\n...\"", decode it
                    if result.startswith('"') and result.endswith('"'):
                        result = json.loads(result)
                        print(f"[AGENT] 📝 Decoded JSON-encoded response")
                except json.JSONDecodeError:
                    # Not JSON, use as-is (might be plain text)
                    result = result.strip('"')
                
                if result and len(result) > 10:  # Basic validation
                    return result
                else:
                    print(f"[AGENT] ⚠️ Empty/short response (attempt {attempt + 1}/{max_retries})")
            else:
                print(f"[AGENT] ❌ API error {response.status_code} (attempt {attempt + 1}/{max_retries})")
            
            # Retry on failure
            if attempt < max_retries - 1:
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
                
        except requests.exceptions.Timeout:
            print(f"[AGENT] ⏱️ Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                continue
        except Exception as e:
            print(f"[AGENT] ❌ LLM error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
    
    print("[AGENT] ❌ All retry attempts failed")
    return ""


# ========== PHASE 1: TEST ANALYSIS ==========
TEST_ANALYSIS_PROMPT = """Analyze the test cases to understand the exact requirements for the evaluator system.

Problem:
{problem}

Code Skeleton:
{skeleton}

Test Cases:
{test_cases}

CRITICAL: The evaluator system will run these tests using unittest framework in a sandboxed environment. Study each test case carefully to understand:

1. **Exact Method Signatures**: What methods must exist? What parameters?
2. **Expected Behavior**: What should each method do? What should it return?
3. **State Management**: What internal state must be maintained?
4. **Edge Cases**: What special cases are tested?
5. **Error Handling**: What exceptions should be raised and when?
6. **Dependencies**: How do objects interact with each other?
7. **Lifecycle**: When are callbacks triggered? When are values updated?
8. **Import Compatibility**: Tests will import from main.py - ensure all required classes/functions are available
9. **Attribute Requirements**: All attributes referenced in tests must exist and be properly initialized
10. **Method Requirements**: All methods called in tests must exist with correct signatures

EVALUATOR SYSTEM NOTES:
- Tests run using unittest framework in isolated environment
- Tests import directly from main.py (no package structure)
- All classes and methods must be available at module level
- Tests expect specific attribute names and method signatures
- Callbacks and state changes must work exactly as expected

For each test, identify:
- What it's testing
- What the expected input/output is
- What internal state changes are required
- What edge cases it covers
- What attributes/methods it expects to exist

Output a detailed test analysis focusing on the exact requirements for the evaluator system."""


# ========== PHASE 2: DESIGN ==========
DESIGN_PROMPT = """Design the complete algorithm based on test requirements for the evaluator system.

Test Analysis:
{test_analysis}

Skeleton:
{skeleton}

EVALUATOR COMPATIBILITY REQUIREMENTS:
- All classes must be defined at module level (no nested classes)
- All methods must be accessible to unittest framework
- All attributes referenced in tests must exist and be properly initialized
- Import statements must be compatible with direct module import
- No external dependencies beyond Python standard library

Design Requirements:
1. **Class Structure**: Exact class names, inheritance, attributes
2. **Method Signatures**: Exact method names, parameters, return types
3. **State Management**: All internal attributes and their purposes
4. **Dependencies**: How objects reference each other
5. **Lifecycle**: When values are computed, when callbacks fire
6. **Error Handling**: What exceptions to raise and when
7. **Edge Cases**: How to handle special cases from tests
8. **Attribute Initialization**: ALL attributes must be initialized in __init__

For each class:
- List ALL attributes (including private ones like _dependent_cells, _callbacks)
- List ALL methods with exact signatures
- Describe the purpose of each attribute and method
- Explain how objects interact (dependencies, updates, callbacks)
- Specify which attributes are initialized in __init__

For each method:
- What it does
- What it returns
- What side effects it has
- When it's called
- What exceptions it might raise
- How it interacts with other objects

CRITICAL: Ensure every attribute and method referenced in the test cases exists and works exactly as expected by the unittest framework.

Be extremely specific about data structures and relationships."""


# ========== PHASE 3: IMPLEMENTATION ==========
IMPL_PROMPT = """Generate ONLY valid Python code - NO explanations, NO markdown, NO comments.

Design:
{design}

Skeleton:
{skeleton}

Test Cases:
{test_cases}

OUTPUT FORMAT - ONLY THIS, NOTHING ELSE:
```python
<YOUR COMPLETE WORKING CODE HERE>
```

EVALUATOR SYSTEM REQUIREMENTS:
- ⚠️ USE ONLY PYTHON STANDARD LIBRARY (no numpy, pandas, external packages)
- All classes must be defined at module level (no nested classes)
- All classes and methods must be importable by unittest framework
- Implement EXACTLY what the tests expect - match method signatures precisely
- Initialize ALL attributes in __init__ (including _dependent_cells, _callbacks, etc.)
- Implement ALL methods (NO pass statements)
- Handle ALL edge cases from the test cases
- Ensure proper object relationships and dependencies
- Valid Python syntax only
- Self-contained code that runs without external dependencies

CRITICAL FOR EVALUATOR COMPATIBILITY:
- Every attribute referenced in tests must exist and be initialized
- Every method called in tests must exist with correct signature
- All classes must be available for direct import from main.py
- Callbacks and state management must work exactly as expected
- No missing functionality that tests depend on

VALIDATION CHECKLIST:
- All classes have required attributes initialized in __init__
- All methods match test expectations exactly
- Dependencies between objects are properly maintained
- Callbacks are triggered at the right times
- Edge cases from tests are handled
- No missing attributes or methods
- Code is compatible with unittest framework
- All test requirements are met

ALLOWED: collections, itertools, math, re, functools, etc. (built-in modules)
FORBIDDEN: numpy, pandas, scipy, requests, any external packages

START YOUR RESPONSE WITH ```python AND END WITH ```
NO TEXT BEFORE OR AFTER THE CODE BLOCK."""


# ========== PHASE 4: VALIDATION ==========
VALIDATION_PROMPT = """Validate the generated code against test requirements.

Generated Code:
{code}

Test Cases:
{test_cases}

Validation Checklist:
1. **Class Structure**: Do all required classes exist with correct names?
2. **Attributes**: Are all required attributes initialized in __init__?
3. **Methods**: Do all required methods exist with correct signatures?
4. **Dependencies**: Are object relationships properly maintained?
5. **Logic**: Does the business logic match test expectations?
6. **Edge Cases**: Are all edge cases from tests handled?
7. **Error Handling**: Are exceptions raised when expected?

For each test case, verify:
- The code can handle the test's input
- The code produces the expected output
- All required attributes and methods exist
- Dependencies are properly maintained

Output detailed validation results."""


# ========== PHASE 5: DEBUG ==========
DEBUG_PROMPT = """Fix the failing tests for the evaluator system. Output ONLY corrected Python code.

Current Code:
{code}

Test Failures:
{test_output}

Original Test Cases:
{test_cases}

EVALUATOR SYSTEM ANALYSIS:
1. **AttributeError Analysis**: What attributes are missing or not initialized?
2. **MethodError Analysis**: What methods are missing or have wrong signatures?
3. **ImportError Analysis**: Are all classes/functions available for import?
4. **Logic Problems**: What business logic is incorrect?
5. **Dependency Issues**: How are object relationships broken?
6. **Callback Issues**: Are callbacks working as expected?
7. **State Management**: Are object states being maintained correctly?

FIX REQUIREMENTS FOR EVALUATOR:
- ⚠️ USE ONLY PYTHON STANDARD LIBRARY (no numpy, pandas, external packages)
- Fix ALL bugs causing test failures
- Ensure ALL required attributes exist and are initialized in __init__
- Implement ALL missing methods with correct signatures
- Fix object relationships and dependencies
- Handle ALL edge cases from original tests
- Ensure code is compatible with unittest framework
- Make sure all classes are available for direct import
- Valid Python syntax only

CRITICAL FIXES:
- Initialize ALL attributes in __init__ methods
- Implement ALL methods referenced in tests
- Fix AttributeError issues by adding missing attributes
- Fix MethodError issues by implementing missing methods
- Ensure callbacks work exactly as expected
- Fix object dependency relationships
- Handle all edge cases from test cases

VALIDATION CHECKLIST:
- All AttributeError issues resolved
- All methods implemented correctly with right signatures
- Dependencies properly maintained
- Callbacks work as expected
- Edge cases handled
- No missing functionality
- Code compatible with unittest framework
- All test requirements met

OUTPUT FORMAT - ONLY THIS:
```python
<YOUR FIXED CODE HERE>
```

NO explanations, NO analysis, NO markdown formatting.
START WITH ```python AND END WITH ```"""


def read_skeleton(repo_dir: str) -> str:
    """Read main.py skeleton if it exists."""
    main_path = os.path.join(repo_dir, "main.py")
    if os.path.exists(main_path):
        with open(main_path, "r") as f:
            return f.read()
    return ""


def read_test_cases(repo_dir: str) -> str:
    """Read test cases if they exist."""
    test_path = os.path.join(repo_dir, "tests.py")
    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            return f.read()
    return ""


def analyze_test_failures(test_output: str) -> dict:
    """Analyze test failures to extract key error patterns."""
    import re
    
    errors = {
        "attribute_errors": [],
        "method_errors": [],
        "logic_errors": [],
        "import_errors": [],
        "syntax_errors": []
    }
    
    # Extract AttributeError patterns
    attr_errors = re.findall(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", test_output)
    for class_name, attr_name in attr_errors:
        errors["attribute_errors"].append(f"{class_name} missing {attr_name}")
    
    # Extract method call errors
    method_errors = re.findall(r"TypeError: '(\w+)' object is not callable", test_output)
    for obj_name in method_errors:
        errors["method_errors"].append(f"{obj_name} is not callable")
    
    # Extract assertion errors
    assertion_errors = re.findall(r"AssertionError: (.+)", test_output)
    for error in assertion_errors:
        errors["logic_errors"].append(error)
    
    return errors


def validate_evaluator_compatibility(code: str, test_cases: str) -> tuple[bool, list[str]]:
    """Validate that code will work with the evaluator system."""
    import re
    
    issues = []
    
    # Check for required classes mentioned in tests
    class_patterns = re.findall(r'class (\w+)', test_cases)
    for class_name in class_patterns:
        if f"class {class_name}" not in code:
            issues.append(f"Missing class: {class_name}")
    
    # Check for required methods mentioned in tests
    method_patterns = re.findall(r'def (test_\w+)', test_cases)
    for method_name in method_patterns:
        # Extract the actual method being tested
        test_content = re.search(rf'def {method_name}\(.*?\):(.*?)(?=def|\Z)', test_cases, re.DOTALL)
        if test_content:
            test_body = test_content.group(1)
            # Look for method calls in the test
            method_calls = re.findall(r'(\w+)\.(\w+)\(', test_body)
            for obj_name, method_name in method_calls:
                if f"def {method_name}" not in code and f"self.{method_name}" not in code:
                    issues.append(f"Missing method: {method_name}")
    
    # Check for attribute access in tests
    attr_patterns = re.findall(r'(\w+)\.(\w+)(?![(])', test_cases)
    for obj_name, attr_name in attr_patterns:
        if f"self.{attr_name}" not in code and f".{attr_name}" not in code:
            issues.append(f"Missing attribute: {attr_name}")
    
    # Check for proper __init__ methods
    class_defs = re.findall(r'class (\w+).*?:', code)
    for class_name in class_defs:
        if f"class {class_name}" in code:
            class_content = re.search(rf'class {class_name}.*?:(.*?)(?=class|\Z)', code, re.DOTALL)
            if class_content and "def __init__" not in class_content.group(1):
                issues.append(f"Missing __init__ method in class: {class_name}")
    
    return len(issues) == 0, issues


def validate_imports(code: str) -> tuple[bool, list[str]]:
    """Check if code only uses standard library imports."""
    import re
    
    # List of forbidden packages
    FORBIDDEN = {
        'numpy', 'np', 'pandas', 'pd', 'scipy', 'sklearn', 'tensorflow', 'tf',
        'torch', 'requests', 'flask', 'django', 'matplotlib', 'seaborn',
        'bs4', 'beautifulsoup', 'selenium', 'scrapy', 'sqlalchemy'
    }
    
    # Find all import statements
    imports = re.findall(r'^\s*(?:from|import)\s+(\w+)', code, re.MULTILINE)
    
    forbidden_found = []
    for imp in imports:
        if imp in FORBIDDEN:
            forbidden_found.append(imp)
    
    return len(forbidden_found) == 0, forbidden_found


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.
    Handles multiple formats: filename.py\n<code>\n, ```python blocks, etc.
    """
    lines = response.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if line is a filename: ends with .py, no spaces, not too long
        if (stripped.endswith('.py') and 
            ' ' not in stripped and 
            3 < len(stripped) < 50 and
            '/' not in stripped and
            not stripped.startswith('#')):
            
            # Found a filename - if we already have content for main.py, that's what we want
            if current_filename == 'main.py' and current_content:
                return '\n'.join(current_content).strip()
            
            # Start tracking new file
            current_filename = stripped
            current_content = []
        else:
            # This is content
            if current_filename:
                current_content.append(line)
    
    # Return content for main.py if found
    if current_filename == 'main.py' and current_content:
        return '\n'.join(current_content).strip()
    
    return None


def extract_python_code_robust(response: str) -> str:
    """
    Robust Python code extraction with multiple strategies.
    """
    import re
    
    # Strategy 1: Look for ```python blocks
    python_blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if python_blocks:
        return python_blocks[0].strip()
    
    # Strategy 2: Look for generic ``` blocks
    generic_blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    if generic_blocks:
        return generic_blocks[0].strip()
    
    # Strategy 3: Look for filename + content format
    extracted = extract_code_from_response(response)
    if extracted:
        return extracted
    
    # Strategy 4: Look for Python keywords and extract from there
    lines = response.strip().split('\n')
    start_idx = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith(('def ', 'class ', 'import ', 'from ')) or
            (stripped.startswith('#') and i < 5)):
            start_idx = i
            break
    
    if start_idx > 0:
        return '\n'.join(lines[start_idx:]).strip()
    
    # Strategy 5: Return the whole response if it looks like Python
    if any(keyword in response for keyword in ['def ', 'class ', 'import ', 'from ']):
        return response.strip()
    
    return None


def write_solution(repo_dir: str, code: str):
    """Write solution to main.py - extracts Python code from LLM response."""
    print(f"[AGENT DEBUG] Received {len(code)} chars from LLM")
    print(f"[AGENT DEBUG] First 300 chars:\n{code[:300]}")
    
    original = code
    
    # Step 0: Decode JSON escaping if present (e.g., "```python\n" -> "```python\n")
    try:
        import json
        # If code starts with quote and contains escaped \n, it might be JSON-encoded
        if code.startswith('"') and '\\n' in code:
            code = json.loads(code)
            print(f"[AGENT DEBUG] 📝 Decoded JSON escaping")
    except:
        pass  # Not JSON-encoded, continue
    
    # Clean common prefixes/suffixes
    code = code.strip()
    
    # Remove ```python or ``` markers (handle both markdown and raw format)
    if code.startswith('```python'):
        code = code[9:].strip()  # Remove ```python
        print(f"[AGENT DEBUG] 🧹 Removed ```python prefix")
    elif code.startswith('```'):
        code = code[3:].strip()  # Remove ```
        print(f"[AGENT DEBUG] 🧹 Removed ``` prefix")
    
    if code.endswith('```'):
        code = code[:-3].strip()  # Remove closing ```
        print(f"[AGENT DEBUG] 🧹 Removed ``` suffix")
    
    # Use robust extraction
    extracted = extract_python_code_robust(code)
    if extracted:
        code = extracted
        print(f"[AGENT DEBUG] ✅ Extracted using robust method: {len(code)} chars")
    else:
        print(f"[AGENT DEBUG] ⚠️ Robust extraction failed, using original")
    
    # Final validation
    if not code or len(code) < 10:
        print(f"[AGENT ERROR] ❌ Extraction failed!")
        print(f"[AGENT ERROR] Original ({len(original)} chars):")
        print("=" * 80)
        print(original[:1000])
        print("=" * 80)
        code = original.strip()
    
    # Validate imports (only standard library)
    valid, forbidden = validate_imports(code)
    if not valid:
        print(f"[AGENT WARNING] ⚠️ Code uses forbidden packages: {', '.join(forbidden)}")
    
    # Write to file
    main_path = os.path.join(repo_dir, "main.py")
    with open(main_path, "w") as f:
        f.write(code)
    
    print(f"[AGENT] ✅ Wrote {len(code)} chars to main.py")
    print(f"[AGENT DEBUG] First 200 chars:")
    print(code[:200])


def run_tests(repo_dir: str) -> tuple[str, bool]:
    """Run tests and return (output, passed)."""
    test_path = os.path.join(repo_dir, "tests.py")
    if not os.path.exists(test_path):
        return "No tests found", True
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout + "\n" + result.stderr
        passed = (result.returncode == 0)
        return output, passed
    except subprocess.TimeoutExpired:
        return "Tests timed out", False
    except Exception as e:
        return str(e), False


def init_git(repo_dir: str):
    """Initialize git repository with proper config and initial commit."""
    git_dir = os.path.join(repo_dir, ".git")
    if os.path.exists(git_dir):
        print("[AGENT] Git already initialized")
        return
    
    try:
        print("[AGENT] Initializing git repository...")
        # Initialize repo
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        
        # Configure git
        subprocess.run(["git", "config", "user.email", "agent@ridges.ai"], 
                      cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Ridges Agent"], 
                      cwd=repo_dir, check=True, capture_output=True)
        
        # Add all Python files
        subprocess.run(["git", "add", "*.py"], cwd=repo_dir, check=True, capture_output=True)
        
        # Create initial commit
        result = subprocess.run(["git", "commit", "-m", "Initial commit"], 
                               cwd=repo_dir, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[AGENT] ✅ Git initialized with initial commit")
        else:
            print(f"[AGENT] ⚠️ Git initialized but commit failed: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"[AGENT] ❌ Git init error: {e}")


def generate_patch(repo_dir: str) -> str:
    """Generate git diff patch."""
    try:
        subprocess.run(["git", "add", "main.py"], cwd=repo_dir, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except Exception as e:
        print(f"[AGENT] Patch generation failed: {e}")
        return ""


def agent_main(input_dict: dict) -> str:
    """
    Main entry point for competitive programming agent.
    
    Args:
        input_dict: Dict with 'problem_statement' key
    
    Returns:
        Git diff patch of the solution
    """
    print("[AGENT] ========================================")
    print("[AGENT] Competitive Programming Agent Started")
    print("[AGENT] ========================================")
    
    problem = input_dict.get("problem_statement", "")
    if not problem:
        print("[AGENT] ERROR: No problem_statement")
        return ""
    
    # Determine repo directory
    # Benchmark framework changes to workspace_dir, so repo is at ./repo
    repo_dir = "./repo" if os.path.exists("./repo") else "."
    
    print(f"[AGENT] Using repo_dir: {repo_dir}")
    print(f"[AGENT] Problem: {len(problem)} chars")
    
    # Read skeleton and test cases
    skeleton = read_skeleton(repo_dir)
    test_cases = read_test_cases(repo_dir)
    print(f"[AGENT] Skeleton: {len(skeleton)} chars")
    print(f"[AGENT] Test cases: {len(test_cases)} chars")
    
    # Initialize git
    init_git(repo_dir)
    
    # ========== PHASE 1: TEST ANALYSIS ==========
    print("\n[AGENT] ===== PHASE 1: TEST ANALYSIS =====")
    test_analysis = call_llm([{
        "role": "user",
        "content": TEST_ANALYSIS_PROMPT.format(problem=problem, skeleton=skeleton, test_cases=test_cases)
    }])
    
    if not test_analysis:
        print("[AGENT] ERROR: Test analysis failed")
        return ""
    
    print(f"[AGENT] Test analysis: {len(test_analysis)} chars")
    
    # ========== PHASE 2: DESIGN ==========
    print("\n[AGENT] ===== PHASE 2: DESIGN =====")
    design = call_llm([{
        "role": "user",
        "content": DESIGN_PROMPT.format(test_analysis=test_analysis, skeleton=skeleton)
    }])
    
    if not design:
        print("[AGENT] ERROR: Design failed")
        return ""
    
    print(f"[AGENT] Design: {len(design)} chars")
    
    # ========== PHASE 3: IMPLEMENTATION ==========
    print("\n[AGENT] ===== PHASE 3: IMPLEMENTATION =====")
    code = call_llm([{
        "role": "user",
        "content": IMPL_PROMPT.format(design=design, skeleton=skeleton, test_cases=test_cases)
    }])
    
    if not code:
        print("[AGENT] ERROR: Implementation failed")
        return ""
    
    print(f"[AGENT] Implementation: {len(code)} chars")
    write_solution(repo_dir, code)
    
    # ========== PHASE 4: VALIDATION ==========
    print("\n[AGENT] ===== PHASE 4: VALIDATION =====")
    
    # Check evaluator compatibility
    is_compatible, compatibility_issues = validate_evaluator_compatibility(code, test_cases)
    if not is_compatible:
        print(f"[AGENT] ⚠️ Evaluator compatibility issues: {compatibility_issues}")
    else:
        print(f"[AGENT] ✅ Code is compatible with evaluator system")
    
    validation = call_llm([{
        "role": "user",
        "content": VALIDATION_PROMPT.format(code=code, test_cases=test_cases)
    }])
    
    if validation:
        print(f"[AGENT] Validation: {len(validation)} chars")
        print(f"[AGENT] Validation results: {validation[:200]}...")
    
    # ========== PHASE 5: SELF-TESTING & DEBUG ==========
    print("\n[AGENT] ===== PHASE 5: SELF-TESTING & DEBUG =====")
    print("[AGENT] Running unit tests against generated code...")
    
    max_iterations = 3
    for iteration in range(max_iterations):
        print(f"\n[AGENT] 🧪 Self-test iteration {iteration + 1}/{max_iterations}")
        
        test_output, passed = run_tests(repo_dir)
        
        if passed:
            print("[AGENT] ✅ All unit tests passed! Code validated.")
            break
        
        if iteration < max_iterations - 1:
            print(f"[AGENT] ❌ Tests failed, analyzing failures and regenerating code...")
            
            # Analyze test failures
            error_analysis = analyze_test_failures(test_output)
            print(f"[AGENT] 🔍 Error analysis: {error_analysis}")
            
            # Read current code
            with open(os.path.join(repo_dir, "main.py"), "r") as f:
                current_code = f.read()
            
            # Debug and fix
            print(f"[AGENT] 🔧 Debugging failed tests...")
            fixed_code = call_llm([{
                "role": "user",
                "content": DEBUG_PROMPT.format(code=current_code, test_output=test_output, test_cases=test_cases)
            }])
            
            if fixed_code:
                print(f"[AGENT] 💾 Writing fixed code...")
                write_solution(repo_dir, fixed_code)
        else:
            print("[AGENT] ⚠️ Max debug iterations reached, returning best attempt")
    
    # ========== GENERATE PATCH ==========
    print("\n[AGENT] ===== GENERATING PATCH =====")
    patch = generate_patch(repo_dir)
    
    print(f"[AGENT] Patch: {len(patch)} chars")
    print("[AGENT] ========================================")
    print("[AGENT] Agent Finished")
    print("[AGENT] ========================================")
    
    return patch
