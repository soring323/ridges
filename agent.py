"""
Dynamic Adaptive Problem Solving Agent
========================================
An intelligent agent that adapts its reasoning chain based on problem type, like human thinking.

Design Philosophy:
- 🧠 Dynamic Reasoning: LLM naturally identifies problem type, not hardcoded categories
- 🔄 Adaptive Chain: Each phase builds on previous understanding
- 🎯 Problem-Agnostic: Works on ANY problem type - algorithms, APIs, patterns, etc.
- 📝 Simple & Clear: ~500 lines vs complex multi-agent frameworks (3000+ lines)
- 🎪 Self-Validating: Tests and debugs its own code before returning

Dynamic Five-Phase Approach:
Phase 0: Problem Type Detection (Open-ended) - "What kind of problem is this?"
Phase 1: Analysis (Adaptive) - Tailored to identified problem type
Phase 2: Design (Adaptive) - Strategy based on problem characteristics
Phase 3: Implementation - Following adaptive design
Phase 4: Self-Testing & Debug - Iterative refinement (up to 3 iterations)

Why Dynamic?
Instead of "if algorithm then X, if REST API then Y", the LLM naturally determines:
- What type of problem it is (in its own words)
- What technical approach is needed
- What patterns and tools to use
→ This allows handling NEW problem types not hardcoded into the agent!

Key Features:
- 🧠 Natural problem understanding: No forced categorization
- 🔄 Adaptive reasoning: Each phase informed by previous
- 🎯 Future-proof: Handles unforeseen problem types
- 🔧 Self-testing: Runs unit tests and auto-debugs (up to 3 iterations)
- 📦 Standard library only: No external dependencies
- 🔄 Retry logic: Robust LLM calls with exponential backoff

Think → Understand → Adapt → Implement → Validate
Like a human brain solving problems naturally.
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
# MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
MODEL = "deepseek-ai/DeepSeek-V3-0324"
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


# ========== PHASE 0: PROBLEM TYPE DETECTION (OPEN-ENDED) ==========
PROBLEM_TYPE_PROMPT = """You are analyzing a programming problem. Think carefully and methodically.

Problem:
{problem}

Code Skeleton:
{skeleton}

CHAIN OF THOUGHT - Think through each step:

Step 1 - Read Carefully:
- What does the problem ask me to build?
- What are the key nouns (objects/entities) mentioned?
- What are the key verbs (actions/behaviors) required?

Step 2 - Identify Domain:
- Is this algorithmic (sorting, graphs, DP, math)?
- Is this reactive/event-driven (callbacks, observers, state changes)?
- Is this data manipulation (JSON, parsing, transformation)?
- Is this OOP-focused (class design, inheritance, patterns)?

Step 3 - Technical Patterns:
- Looking at the skeleton, what methods need to be implemented?
- Do I see patterns like: observer, state machine, builder, factory?
- What standard library would naturally fit? (collections, json, re, itertools, etc.)

Step 4 - Key Challenge:
- What's the HARD part of this problem?
- What could go wrong if I'm not careful?

OUTPUT (2-3 sentences):
Describe the problem type, domain, key technical approach, and main challenge."""


# ========== PHASE 1: ANALYSIS (ADAPTIVE) ==========
ANALYSIS_PROMPT = """Analyze this programming problem in depth.

Problem Type (from your earlier analysis):
{problem_type}

Problem Statement:
{problem}

Code Skeleton:
{skeleton}

CHAIN OF THOUGHT - Analyze systematically:

STEP 1 - List ALL Requirements:
Read the problem statement line by line. What MUST the solution do?
- List each requirement explicitly
- Include both explicit AND implicit requirements
- What behaviors are described?

STEP 2 - Method Signatures (from skeleton):
For EACH method in the skeleton:
- Method name: ___
- Input parameters: ___ (types and meaning)
- Return value: ___ (type and meaning)
- Purpose: ___

STEP 3 - State Management:
- What data must persist between method calls?
- What data can be computed on-the-fly?
- Best data structure: list, dict, set, custom class?
- Why is this structure optimal?

STEP 4 - Technical Approach:
- What design pattern fits? (Observer, State Machine, Builder, etc.)
- What standard library modules help? (collections, json, re, etc.)
- Any special handling needed? (callbacks, JSON, parsing, etc.)

STEP 5 - Edge Cases & Pitfalls:
Standard edge cases:
- What if inputs are empty or None?
- What if inputs are invalid?
- What boundary conditions exist?

Pattern-specific edge cases (discover through CONCRETE EXAMPLES):
- What design pattern did I identify in Step 4?

Now think through a CONCRETE scenario for edge cases:
1. Create a simple example with this pattern
2. Walk through what happens step-by-step
3. Identify where things could go wrong

Example walkthrough template:
- Initial state: ___ (describe starting values)
- Action 1: ___ (what happens)
- Intermediate state: ___ (what changes)
- Action 2: ___ (what happens next)
- Final state: ___ (result)

Now ask yourself:
- In this scenario, what SHOULD trigger notifications/updates/changes?
- What SHOULD NOT trigger them, even though something changed?
- Example: If a dependency changes but the computed result stays the same, what happens?
- Example: If I update from value A → B → A (back to original), what should happen?

The key question:
- Am I reacting to INPUTS changing, or OUTPUTS changing?
- How do I distinguish between the two?

DOMAIN-SPECIFIC PATTERNS (based on problem type):

If this is a REACTIVE/OBSERVER/SPREADSHEET system:
⚠️ Common edge case: Dependencies change but computed value stays the same
- When implementing notification logic, ask yourself:
  * Should callbacks fire EVERY TIME a dependency changes?
  * Or ONLY when the computed result actually changes?
- How do you detect if a result changed? (Hint: need before & after values to compare)

If this is a STATE MACHINE:
⚠️ Common edge case: Invalid state transitions
- When implementing transitions, ask: Is every transition valid, or only certain ones?
- Do I need to check validity before changing state?

If this is a PARSER/INTERPRETER:
⚠️ Common edge case: Edge cases in delimiters, empty strings, malformed input
- What if input is empty? What if delimiters are missing?

If this is a GRAPH/TREE problem:
⚠️ Common edge case: Cycles, disconnected components, empty graph
- Need to handle cycles, ensure termination

How do I avoid these pitfalls?

SELF-CHECK:
- Did I understand ALL requirements?
- Am I clear on what each method should do?
- Do I know what data structures to use?

OUTPUT: Comprehensive analysis covering all 5 steps above."""


# ========== PHASE 1.5: TEST EXPECTATIONS ANALYSIS (DYNAMIC) ==========
TEST_SIGNATURE_PROMPT = """Analyze the test code to understand what the implementation must provide.

Test Code:
{test_code}

Analyze and extract:

1. **Method Signatures**: What methods are being called? What are their EXACT parameters?
   - Look at how tests instantiate classes and call methods
   - Note parameter names, types, and whether they're optional

2. **Callback Patterns**: Are there callbacks, observers, or event handlers?
   - If yes, what are their signatures?
   - How many arguments do they receive?

2.5. **Parameter Data Structures**: For each parameter passed to __init__ or methods, what is its exact structure?
   - Look at actual test values: copy them exactly as they appear
   - What is the nested structure? (lists, dicts, primitives, etc.)
   - Show examples from multiple tests to reveal the pattern
   - Based on the structure, how should the code access/iterate these values?

3. **Return Value Expectations**: What do tests expect methods to return?
   - Look at the ACTUAL VALUES in assertEqual(), assertDictEqual(), etc.
   - What TYPE is expected? (int, str, list, dict, etc.)
   - Are there any PATTERNS in the expected values?
   - If strings/lists of strings: Examine the EXACT characters used (don't just read, look closely)

4. **Value Format Discovery**: Compare what problem describes vs what tests expect
   - Do problem descriptions use one format but tests expect another?
   - If problem shows certain symbols/characters, what do tests actually use?
   - Are the characters IDENTICAL or just similar-looking?
   - Look for discrepancies between problem statement units/formats and test expectations

5. **Input Handling**: What formats/types do tests pass as input?
   - Look at test data - what variations are tested?
   - Are there edge cases in the input formats?

6. **Special Behaviors**: Any decorators, properties, magic methods, or special patterns?

7. **Callback/Observer Edge Cases** (if callbacks exist):
   For EACH test with "not", "shouldn't", "only", or conditional language:
   
   Step A: List the test name
   Step B: Break down what the test name says:
     - What behavior is being tested?
     - What condition must be met?
     - What should happen vs NOT happen?
   
   Step C: Derive the implementation requirement:
     - What must my code CHECK before doing something?
     - What information do I need to make that check?
   
   Breakdown:
   - "callbacks" = testing callback notification
   - "should_not_be_called" = callbacks must NOT fire
   - "if_dependencies_change" = even though inputs changed
   - "but_output_value_doesn't_change" = when result stays same
   
   Implementation requirement:
   → Before notifying callbacks, CHECK if output value actually changed
   → Need: old_output and new_output to compare
   → Logic: if old != new, THEN notify
   
   Do this analysis for ALL conditional tests!

⚠️ CRITICAL THINKING:
- Examine ACTUAL VALUES in test assertions - they reveal the truth!
- Read TEST NAMES - they describe EXACT behavior expected
- If problem says one thing but tests expect another, trust the tests
- Look for patterns across multiple test cases
- Negative tests (should NOT happen) are as important as positive tests
- Only mention requirements you can DIRECTLY OBSERVE in the tests

Be specific and evidence-based. Quote test lines if helpful."""


# ========== PHASE 2: DESIGN (ADAPTIVE) ==========
DESIGN_PROMPT = """Design the complete solution.

Problem Type:
{problem_type}

Analysis:
{analysis}

Test Signatures (CRITICAL - match these exactly):
{test_signatures}

Skeleton:
{skeleton}

CHAIN OF THOUGHT - Design step-by-step:

STEP 1 - Class Attributes (Instance Variables):
What data needs to be stored in __init__?
- List each attribute: self.___ = ___
- What is its type? (list, dict, set, etc.)
- What is its initial value?
- Why do we need it?

STEP 2 - Method Design (One by one):
For EACH method in the skeleton, design its logic:

Method: ___
Inputs: ___
Output: ___
Algorithm:
  1. First, do ___
  2. Then, do ___
  3. Finally, return ___
State changes: What attributes get modified?

STEP 3 - Data Flow:
Walk through a concrete example:
- Start: Initial state is ___
- Call method1(args) → state becomes ___
- Call method2(args) → state becomes ___
- Result: ___

Now walk through an EDGE CASE scenario:
- What if I do the same operation twice?
- What if an input changes but computed output doesn't?
- What if I go back to a previous state?
- In each case: What SHOULD happen vs what MIGHT happen if I implement naively?

STEP 4 - Edge Cases:
For each edge case from analysis, design the handling:
- Edge case: ___
- How to handle: ___
- Where to check: in method ___

For pattern-specific edge cases (based on YOUR pattern):
- If my pattern involves notifications/callbacks/observers:
  * What condition determines whether to notify?
  * Do I need to compare something before notifying?
  * What information do I need to make this comparison?
- If my pattern involves state/dependencies:
  * What triggers an update?
  * Should ALL changes trigger updates, or only CERTAIN changes?
  * How do I distinguish between the two?

REMINDER - Domain-Specific Design Patterns:

For REACTIVE/OBSERVER systems:
- Design the notification logic carefully:
  1. Store current value BEFORE recomputing
  2. Compute new value
  3. Compare old vs new
  4. Only notify if different
- Why? Input changes don't always mean output changes

For STATE MACHINES:
- Design transition validation:
  1. Check if transition is valid
  2. If valid, change state
  3. If invalid, raise error or ignore

For CACHING systems:
- Design invalidation logic:
  1. Check if cached value is still valid
  2. If invalid, recompute
  3. Store and return new value

STEP 5 - Critical Details:
⚠️ VERIFY these match test expectations:
- Method signatures: Do parameter names/types match tests?
- Return types: Do return values match test assertions?
- Callback signatures: If tests pass callbacks, how many args do they take?

SELF-CHECK before moving to implementation:
✓ Did I design ALL methods from skeleton?
✓ Are ALL attributes initialized in __init__?
✓ Did I handle ALL edge cases?
✓ Do signatures match test expectations?

OUTPUT: Complete design covering all steps above, ready to implement."""


# ========== PHASE 3: IMPLEMENTATION ==========
IMPL_PROMPT = """Generate ONLY valid Python code - NO explanations, NO markdown, NO comments.

Design:
{design}

Test Signatures (MATCH EXACTLY):
{test_signatures}

Skeleton:
{skeleton}

OUTPUT FORMAT - ONLY THIS, NOTHING ELSE:
```python
<YOUR COMPLETE WORKING CODE HERE>
```

CRITICAL REQUIREMENTS:
- ⚠️ USE ONLY PYTHON STANDARD LIBRARY (no numpy, pandas, external packages)
- ⚠️ MATCH FUNCTION SIGNATURES EXACTLY (wrong signatures = TypeError)
- Store ALL parameters in __init__
- Initialize ALL data structures
- Implement ALL methods (NO pass statements)
- Handle edge cases
- Valid Python syntax only
- Self-contained code that runs without external dependencies

ALLOWED IMPORTS: 
- Standard library only: collections, itertools, math, re, functools, json, etc.

FORBIDDEN: 
- numpy, pandas, scipy, requests, flask, django, any external packages

BEFORE YOU CODE - VERIFY YOUR UNDERSTANDING:
1. ✓ I know EXACTLY what each method should do
2. ✓ I know EXACTLY what data structures to use
3. ✓ I know EXACTLY what to return from each method
4. ✓ I've checked the test signatures - parameter names match
5. ✓ I've checked the test signatures - return types match
6. ✓ I know how to handle edge cases (empty inputs, invalid data, etc.)

IMPLEMENTATION CHECKLIST (verify as you code):
□ __init__: Store ALL constructor parameters as instance variables
□ __init__: Initialize ALL data structures (lists, dicts, etc.) to proper initial values
□ Each method: Implement the EXACT algorithm from the design
□ Each method: Use the EXACT parameter names from the skeleton
□ Each method: Return the EXACT type expected by tests
□ Edge cases: Add checks for empty/invalid inputs
□ State: Update instance variables correctly when state changes
□ Standard library only: No numpy, pandas, requests, etc.
□ Callbacks/notifications: Did I implement the correct trigger condition from my design?

COMMON MISTAKES TO AVOID:
- Don't change parameter names from skeleton
- Don't forget to store constructor parameters
- Don't return wrong type (e.g., string instead of list)
- Don't use external libraries
- Don't leave any method with just 'pass'

START YOUR RESPONSE WITH ```python AND END WITH ```
NO TEXT BEFORE OR AFTER THE CODE BLOCK."""


# ========== PHASE 4: DEBUG ==========
DEBUG_PROMPT = """Fix the failing tests. Output ONLY corrected Python code.

Original Problem Statement:
{problem}

Current Code:
{code}

Test Failures:
{test_output}

Expected Test Signatures (match these exactly):
{test_signatures}

CHAIN OF THOUGHT - Debug systematically:

STEP 1 - Read the Error Carefully:
- What TYPE of error? (TypeError, AssertionError, AttributeError, etc.)
- Which test failed? What does the test name tell me?
- What is EXPECTED vs ACTUAL? Write them down:
  Expected: ___
  Actual: ___

STEP 2 - Identify the Root Cause:
Ask yourself:
- Is this a TYPE error? (returning wrong type - str vs list, int vs str, etc.)
- Is this a LOGIC error? (wrong algorithm, wrong calculation)
- Is this a SIGNATURE error? (wrong parameter names, wrong number of parameters)
- Is this a STATE error? (not updating/storing data correctly)
- Is this an EDGE CASE? (empty input, None, boundary value)

STEP 3 - Find WHERE the Bug Is:
- Which METHOD is causing the failure?
- Which LINE in that method is wrong?
- What is the code doing vs what should it do?

STEP 4 - Plan the Fix:
- What EXACTLY needs to change?
- Will this fix break anything else?
- Does this fix ALL similar failures or just one?

STEP 5 - Verify the Fix:
Before outputting code, check:
✓ Did I fix the root cause, not just the symptom?
✓ Did I keep all working code unchanged?
✓ Will this pass ALL tests, not just the failing one?

COMMON BUG PATTERNS:
- Forgot to store parameter in __init__
- Returning wrong type (str instead of list, etc.)
- Not updating state correctly
- Wrong callback signature (wrong number of parameters)
- Off-by-one error in loops/indexing
- Not handling empty/None inputs

REQUIREMENTS:
- ⚠️ USE ONLY PYTHON STANDARD LIBRARY (no numpy, pandas, external packages)
- Fix the bugs causing test failures
- Keep working code unchanged
- Valid Python syntax only

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


def read_tests(repo_dir: str) -> str:
    """Read tests.py if it exists."""
    test_path = os.path.join(repo_dir, "tests.py")
    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            return f.read()
    return ""


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.
    Handles format: filename.py\n<code>\n
    Inspired by reference implementation's extract_and_write_files.
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
    
    # Strategy 1: Try filename + content format (main.py\n<code>)
    extracted = extract_code_from_response(code)
    if extracted:
        code = extracted
        print(f"[AGENT DEBUG] ✅ Extracted from filename+content format: {len(code)} chars")
    else:
        # Strategy 2: Try ```python blocks (in case still nested)
        import re
        python_blocks = re.findall(r'```python\s*\n(.*?)```', code, re.DOTALL)
        
        if python_blocks:
            code = python_blocks[0].strip()
            print(f"[AGENT DEBUG] ✅ Extracted from nested ```python block: {len(code)} chars")
        else:
            # Strategy 3: Try generic ``` blocks
            generic_blocks = re.findall(r'```\s*\n(.*?)```', code, re.DOTALL)
            if generic_blocks:
                code = generic_blocks[0].strip()
                print(f"[AGENT DEBUG] ✅ Extracted from nested ``` block: {len(code)} chars")
            else:
                # Strategy 4: Look for Python keywords
                print(f"[AGENT DEBUG] 🔍 Looking for Python keywords...")
                lines = code.strip().split("\n")
                
                start_idx = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if (stripped.startswith(('def ', 'class ', 'import ', 'from ')) or
                        (stripped.startswith('#') and i < 5)):
                        start_idx = i
                        print(f"[AGENT DEBUG] Found Python code at line {i}")
                        break
                
                code = "\n".join(lines[start_idx:]).strip()
                print(f"[AGENT DEBUG] ✅ Extracted from line {start_idx}: {len(code)} chars")
    
    # Final validation
    if not code or len(code) < 10:
        print(f"[AGENT ERROR] ❌ Extraction failed!")
        print(f"[AGENT ERROR] Original ({len(original)} chars):")
        print("=" * 80)
        print(original[:1000])
        print("=" * 80)
        code = original.strip()
    
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
    Main entry point for dynamic adaptive problem solving agent.
    
    Dynamic 5-Phase Approach:
    0. Problem Type Detection (Open-ended) - LLM naturally identifies problem type
    1. Analysis (Adaptive) - Tailored to identified type
    2. Design (Adaptive) - Strategy based on problem characteristics
    3. Implementation - Following adaptive design
    4. Self-Testing & Debug - Iterative refinement
    
    This approach allows the agent to handle ANY problem type, including ones
    not explicitly programmed, by letting the LLM naturally understand and adapt.
    
    Args:
        input_dict: Dict with 'problem_statement' key
    
    Returns:
        Git diff patch of the solution
    """
    print("[AGENT] ================================================")
    print("[AGENT] 🧠 Dynamic Adaptive Problem Solving Agent Started")
    print("[AGENT] ================================================")
    
    problem = input_dict.get("problem_statement", "")
    if not problem:
        print("[AGENT] ERROR: No problem_statement")
        return ""
    
    # Determine repo directory
    # Benchmark framework changes to workspace_dir, so repo is at ./repo
    repo_dir = "./repo" if os.path.exists("./repo") else "."
    
    print(f"[AGENT] Using repo_dir: {repo_dir}")
    print(f"[AGENT] Problem: {len(problem)} chars")
    
    # Read skeleton
    skeleton = read_skeleton(repo_dir)
    print(f"[AGENT] Skeleton: {len(skeleton)} chars")
    
    # Read tests
    test_code = read_tests(repo_dir)
    print(f"[AGENT] Tests: {len(test_code)} chars")
    
    # Initialize git
    init_git(repo_dir)
    
    # ========== PHASE 0: PROBLEM TYPE DETECTION (DYNAMIC) ==========
    print("\n[AGENT] ===== PHASE 0: PROBLEM TYPE DETECTION =====")
    print("[AGENT] 🧠 Letting LLM naturally identify the problem type...")
    problem_type = call_llm([{
        "role": "user",
        "content": PROBLEM_TYPE_PROMPT.format(problem=problem, skeleton=skeleton)
    }])
    
    if not problem_type:
        print("[AGENT] ERROR: Problem type detection failed")
        return ""
    
    print(f"[AGENT] ✅ Problem Type Identified: {len(problem_type)} chars")
    print(f"[AGENT] Preview: {problem_type[:200]}...")
    
    # ========== PHASE 1: ANALYSIS (ADAPTIVE) ==========
    print("\n[AGENT] ===== PHASE 1: ANALYSIS (ADAPTIVE) =====")
    print("[AGENT] 🎯 Tailoring analysis based on identified problem type...")
    analysis = call_llm([{
        "role": "user",
        "content": ANALYSIS_PROMPT.format(problem_type=problem_type, problem=problem, skeleton=skeleton)
    }])
    
    if not analysis:
        print("[AGENT] ERROR: Analysis failed")
        return ""
    
    print(f"[AGENT] Analysis: {len(analysis)} chars")
    
    # ========== PHASE 1.5: TEST EXPECTATIONS ANALYSIS (DYNAMIC) ==========
    print("\n[AGENT] ===== PHASE 1.5: TEST EXPECTATIONS ANALYSIS =====")
    print("[AGENT] 🔍 Discovering what tests reveal about requirements...")
    if test_code:
        test_signatures = call_llm([{
            "role": "user",
            "content": TEST_SIGNATURE_PROMPT.format(test_code=test_code)
        }])
        print(f"[AGENT] Test Expectations: {len(test_signatures)} chars")
    else:
        print("[AGENT] ⚠️ No tests found, skipping expectations analysis")
        test_signatures = "No tests available"
    
    # ========== PHASE 2: DESIGN (ADAPTIVE) ==========
    print("\n[AGENT] ===== PHASE 2: DESIGN (ADAPTIVE) =====")
    print("[AGENT] 📐 Creating design tailored to problem type...")
    design = call_llm([{
        "role": "user",
        "content": DESIGN_PROMPT.format(
            problem_type=problem_type,
            analysis=analysis,
            test_signatures=test_signatures,
            skeleton=skeleton
        )
    }])
    
    if not design:
        print("[AGENT] ERROR: Design failed")
        return ""
    
    print(f"[AGENT] Design: {len(design)} chars")
    
    # ========== PHASE 3: IMPLEMENTATION ==========
    print("\n[AGENT] ===== PHASE 3: IMPLEMENTATION =====")
    code = call_llm([{
        "role": "user",
        "content": IMPL_PROMPT.format(
            design=design,
            test_signatures=test_signatures,
            skeleton=skeleton
        )
    }])
    
    if not code:
        print("[AGENT] ERROR: Implementation failed")
        return ""
    
    print(f"[AGENT] Implementation: {len(code)} chars")
    write_solution(repo_dir, code)
    
    # ========== PHASE 4: SELF-TESTING & DEBUG ==========
    print("\n[AGENT] ===== PHASE 4: SELF-TESTING & DEBUG =====")
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
            
            # Read current code
            with open(os.path.join(repo_dir, "main.py"), "r") as f:
                current_code = f.read()
            
            # Debug and fix
            print(f"[AGENT] 🔧 Debugging failed tests...")
            fixed_code = call_llm([{
                "role": "user",
                "content": DEBUG_PROMPT.format(
                    problem=problem,
                    code=current_code,
                    test_output=test_output,
                    test_signatures=test_signatures
                )
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
