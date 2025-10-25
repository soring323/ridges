"""
Test-Driven Agent with Tree-of-Thoughts (ToT)
==============================================
Enhanced version using Tree-of-Thoughts for architecture exploration.

Key Enhancements over CoT:
1. Generate multiple architecture candidates simultaneously
2. Evaluate each candidate independently
3. Score and rank architectures before implementation
4. Explore multiple solution branches in parallel
5. Select best architecture based on reasoning scores

ToT Strategy:
- Generate 3-5 candidate architectures
- Evaluate each for: correctness, simplicity, testability
- Score each candidate (0-10)
- Implement top-ranked architecture
- Fallback to next best if top fails
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

# Model selection
REASONING_MODEL = "deepseek-ai/DeepSeek-V3-0324"
CODING_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
FAST_MODEL = "deepseek-ai/DeepSeek-V3-0324"

MAX_ITERATIONS = 10
MAX_FIX_STEPS = 100
TOT_CANDIDATES = 10  # Number of architecture candidates to generate (increased for better coverage)

print(f"[AGENT] Test-Driven Agent (ToT) initialized - RUN_ID: {RUN_ID}")

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
                return result
            
            print(f"[NETWORK] HTTP {response.status_code}: {response.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[NETWORK] Timeout (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"[NETWORK] Error: {e} (attempt {attempt + 1}/{max_retries})")
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts")

def are_architectures_similar(name1: str, name2: str) -> bool:
    """Check if two architecture names are similar based on common words."""
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())
    common_words = words1 & words2
    
    if len(common_words) >= 2:
        print(f"[SIMILARITY] '{name1}' vs '{name2}': {len(common_words)} common words")
        return True
    
    return False

# ============================================================================
# Tree-of-Thoughts Architecture Generation
# ============================================================================

def generate_architecture_candidates_tot(
    problem_statement: str,
    test_output: str,
    current_files: Dict[str, str],
    test_results: Dict,
    num_candidates: int = TOT_CANDIDATES,
    avoid_architectures: List[str] = None
) -> List[Dict]:
    """Generate multiple architecture candidates using Tree-of-Thoughts.
    
    Args:
        problem_statement: The problem description
        test_output: Test execution output
        current_files: Current implementation files
        test_results: Parsed test results
        num_candidates: Number of candidates to generate
        avoid_architectures: List of architecture names already tried (to avoid duplicates)
    
    Returns list of candidates, each with:
    - name: Architecture name
    - description: Brief description
    - score: Evaluation score (0-10)
    - reasoning: Why this architecture might work
    """
    
    if avoid_architectures is None:
        avoid_architectures = []
    
    failed_tests = '\n'.join(f"  - {t}" for t in test_results.get('failed_tests', [])[:5])
    error_details = '\n'.join(
        f"  {i+1}. {e['test']}: {e['error'][:200]}"
        for i, e in enumerate(test_results.get('error_details', [])[:3])
    )
    
    prompt = f"""You are an expert software architect analyzing a stuck problem.

## Current Situation:
Problem: {problem_statement}

Tests passing: {test_results['passed']}/{test_results['total']}
Tests failing: {len(test_results.get('failed_tests', []))}

Failed tests:
{failed_tests}

Error details:
{error_details}

## Task - Tree-of-Thoughts Architecture Generation:

Generate {num_candidates} DIFFERENT architectural approaches to solve this problem.
For each approach, provide:
1. **Name**: Short descriptive name
2. **Description**: 2-3 sentence explanation
3. **Score**: Rate 0-10 based on:
   - Correctness: Will it fix the failing tests?
   - Simplicity: Is it easy to implement?
   - Robustness: Will it handle edge cases?
4. **Reasoning**: Why this approach might succeed

{"## Architectures to AVOID (already tried and failed):" if avoid_architectures else ""}
{chr(10).join(f"- {arch}" for arch in avoid_architectures) if avoid_architectures else ""}

Think divergently - explore different paradigms:
- Data structure changes
- Algorithm changes  
- Design pattern changes
- State management changes
- Control flow changes
- Event handling patterns
- Caching/memoization strategies

**CRITICAL**: Generate {num_candidates} COMPLETELY DIFFERENT approaches. Avoid similar patterns to failed architectures above.

Output format:
```
CANDIDATE 1:
Name: [Architecture Name]
Description: [2-3 sentences]
Score: [0-10]
Reasoning: [Why this works]

CANDIDATE 2:
...
```

Generate {num_candidates} diverse candidates now:"""

    try:
        print(f"[ToT] Generating {num_candidates} architecture candidates...")
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,
            temperature=0.9  # High temperature for diversity
        )
        
        # Parse candidates
        candidates = []
        candidate_blocks = re.split(r'CANDIDATE \d+:', response)[1:]  # Skip first empty split
        
        for i, block in enumerate(candidate_blocks[:num_candidates]):
            try:
                # Extract fields
                name_match = re.search(r'Name:\s*(.+)', block)
                desc_match = re.search(r'Description:\s*(.+?)(?=Score:|$)', block, re.DOTALL)
                score_match = re.search(r'Score:\s*(\d+)', block)
                reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=CANDIDATE|$)', block, re.DOTALL)
                
                if name_match and score_match:
                    candidate = {
                        'name': name_match.group(1).strip(),
                        'description': desc_match.group(1).strip() if desc_match else "",
                        'score': int(score_match.group(1)),
                        'reasoning': reasoning_match.group(1).strip() if reasoning_match else ""
                    }
                    
                    # Check for duplicates within this batch
                    is_duplicate_in_batch = False
                    for existing in candidates:
                        if are_architectures_similar(candidate['name'], existing['name']):
                            print(f"[ToT] Candidate {i+1}: '{candidate['name']}' is duplicate of '{existing['name']}', skipping")
                            is_duplicate_in_batch = True
                            break
                    
                    # Check against avoided architectures
                    is_duplicate_avoided = False
                    if avoid_architectures:
                        for avoided in avoid_architectures:
                            if are_architectures_similar(candidate['name'], avoided):
                                print(f"[ToT] Candidate {i+1}: '{candidate['name']}' similar to avoided '{avoided}', skipping")
                                is_duplicate_avoided = True
                                break
                    
                    if not is_duplicate_in_batch and not is_duplicate_avoided:
                        candidates.append(candidate)
                        print(f"[ToT] Candidate {i+1}: {candidate['name']} (score: {candidate['score']}/10)")
            except Exception as e:
                print(f"[ToT] Failed to parse candidate {i+1}: {e}")
                continue
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"[ToT] Kept {len(candidates)} unique candidates after deduplication")
        
        return candidates
        
    except Exception as e:
        print(f"[ToT] Failed to generate candidates: {e}")
        return []

def implement_architecture_tot(
    problem_statement: str,
    candidate: Dict,
    current_files: Dict[str, str]
) -> Optional[Dict[str, str]]:
    """Implement a specific architecture candidate."""
    
    current_code = '\n\n'.join(f"# {fname}\n{code}" for fname, code in current_files.items())
    
    prompt = f"""You are implementing a specific architectural approach to fix a failing test.

## Problem:
{problem_statement}

## Architecture to Implement:
**Name**: {candidate['name']}
**Description**: {candidate['description']}
**Reasoning**: {candidate['reasoning']}

## Current Code:
{current_code}

## Task:
Rewrite the COMPLETE code implementing this architecture. Focus on the key insight from the reasoning.

**CRITICAL**: Output ONLY the code, nothing else. Use this EXACT format:

main.py
```python
# Your complete implementation here
class InputCell:
    ...
```

Do NOT add explanations, markdown headers, or commentary. Just: filename, code block, done.

Generate the complete implementation now:"""

    try:
        print(f"[ToT] Implementing: {candidate['name']}")
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,
            temperature=0.3
        )
        
        print(f"[ToT] Got response ({len(response)} chars)")
        
        # Extract code blocks - try multiple patterns
        code_pattern = r'```(?:python)?\n(.*?)```'
        code_matches = re.findall(code_pattern, response, re.DOTALL)
        
        if code_matches:
            # Concatenate all code blocks
            code = '\n\n'.join(match.strip() for match in code_matches)
        else:
            code = response.strip()
        
        print(f"[ToT] Extracted code ({len(code)} chars)")
        
        # Parse files - more flexible approach
        files = {}
        
        # Method 1: Look for filename.py patterns
        lines = code.split('\n')
        current_file = None
        current_content = []
        
        for line in lines:
            stripped = line.strip()
            # More flexible filename detection
            if stripped.endswith('.py') and len(stripped) < 100:
                if current_file and current_content:
                    content = '\n'.join(current_content).strip()
                    if content and len(content) > 10:  # Must have some content
                        files[current_file] = content
                current_file = stripped
                current_content = []
            elif current_file:
                current_content.append(line)
        
        # Save last file
        if current_file and current_content:
            content = '\n'.join(current_content).strip()
            if content and len(content) > 10:
                files[current_file] = content
        
        # Method 2: If no files found, treat entire code as main.py
        if not files and code.strip():
            if 'def ' in code or 'class ' in code or 'import ' in code:
                files['main.py'] = code.strip()
        
        print(f"[ToT] Parsed {len(files)} file(s): {list(files.keys())}")
        
        if files:
            # Add architecture name to metadata
            files['__architecture_name__'] = candidate['name']
            return files
        
        print(f"[ToT] No files parsed from response")
        print(f"[ToT] First 500 chars of response: {response[:500]}")
        return None
        
    except Exception as e:
        print(f"[ToT] Implementation failed: {e}")
        traceback.print_exc()
        return None

# Import remaining functions from base agent
print("[ToT] Loading base agent functions...")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from test_driven_agent import (
        write_files,
        run_tests,
        parse_test_results,
        generate_solution,
        generate_tests,
        fix_test_failures,
        get_git_diff
    )
except ImportError as e:
    print(f"[ERROR] Failed to import from base agent: {e}")
    print("[ERROR] Make sure test_driven_agent.py is in the same directory")
    sys.exit(1)

# ============================================================================
# ToT-Enhanced CREATE Mode
# ============================================================================

def create_mode_tot(problem_statement: str, tests_path: str) -> str:
    """CREATE mode with Tree-of-Thoughts architecture exploration."""
    
    print("\n" + "="*80)
    print("CREATE MODE - Test-Driven Development (ToT)")
    print("="*80)
    
    # Step 1: Generate initial solution
    print("\n[STEP 1] Generating initial solution...")
    solution_files = generate_solution(problem_statement)
    
    if not solution_files:
        print("[ERROR] Could not generate initial solution")
        return ""
    
    print(f"[STEP 1] Created {len(solution_files)} solution files")
    
    # Step 2: Generate tests
    print("\n[STEP 2] Generating test suite...")
    test_files = generate_tests(problem_statement, solution_files)
    write_files(test_files)
    print(f"[STEP 2] Created {len(test_files)} test files")
    
    # Baseline test
    print("[STEP 2] Running initial tests to establish baseline...")
    success, output = run_tests()
    test_results = parse_test_results(output)
    print(f"[STEP 2] Baseline: {test_results['passed']}/{test_results['total']} tests passing")
    print(f"[STEP 2] Test coverage: {test_results['passed']} passing, {len(test_results.get('failed_tests', []))} failing")
    
    # Step 3: Test-driven refinement with ToT
    print("\n[STEP 3] Test-driven refinement with ToT...")
    
    iteration = 0
    previous_failed = set()
    stuck_count = 0
    alternatives_tried = 0
    max_alternatives = 10
    tried_architectures = {}
    best_solution = solution_files.copy()
    best_score = test_results['passed']
    
    # ToT: Keep track of candidate queue
    candidate_queue = []
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
        
        # Run tests
        success, output = run_tests()
        test_results = parse_test_results(output)
        
        print(f"[TESTS] {test_results['passed']}/{test_results['total']} passed, {len(test_results.get('failed_tests', []))} failed")
        
        if test_results['passed'] > 0:
            passed_tests = test_results.get('passed_tests', [])
            print(f"[TESTS] ✓ Passed: {', '.join(passed_tests[:5])}" + 
                  (f" (+{len(passed_tests)-5} more)" if len(passed_tests) > 5 else ""))
        
        if test_results['failed_tests']:
            print(f"[TESTS] ✗ Failed: {', '.join(test_results['failed_tests'][:5])}" +
                  (f" (+{len(test_results['failed_tests'])-5} more)" if len(test_results['failed_tests']) > 5 else ""))
        
        if test_results.get('error_details'):
            print("[ERRORS] Sample failures:")
            for i, error in enumerate(test_results['error_details'][:3]):
                print(f"  {i+1}. {error['test']}")
                for line in error['error'].split('\n')[:2]:
                    if line.strip():
                        print(f"     {line.strip()}")
        
        if success:
            print("[SUCCESS] All tests passed!")
            break
        
        # Check if stuck
        current_failed = set(test_results['failed_tests'])
        if current_failed == previous_failed and len(current_failed) > 0:
            stuck_count += 1
            
            if stuck_count >= 2 and alternatives_tried < max_alternatives:
                print(f"\n[STUCK] Same {len(current_failed)} test(s) failing for {stuck_count} iterations")
                
                # ToT: Generate candidates if queue is empty
                if not candidate_queue:
                    # Pass already-tried architectures to avoid duplicates
                    failed_archs = list(tried_architectures.keys())
                    if failed_archs:
                        print(f"[ToT] Avoiding {len(failed_archs)} previously tried architecture(s)")
                    
                    print(f"[ToT] Generating {TOT_CANDIDATES} architecture candidates...")
                    candidates = generate_architecture_candidates_tot(
                        problem_statement,
                        output,
                        solution_files,
                        test_results,
                        num_candidates=TOT_CANDIDATES,
                        avoid_architectures=failed_archs
                    )
                    
                    if candidates:
                        print(f"[ToT] Generated {len(candidates)} candidates, ranked by score:")
                        for i, c in enumerate(candidates):
                            print(f"  {i+1}. {c['name']} (score: {c['score']}/10)")
                        candidate_queue = candidates
                    else:
                        print("[ToT] Failed to generate candidates, falling back to regular mode")
                
                # ToT: Try next candidate from queue
                if candidate_queue:
                    candidate = candidate_queue.pop(0)
                    
                    # Check if similar to tried architectures
                    is_duplicate = False
                    for tried_name in tried_architectures.keys():
                        if are_architectures_similar(candidate['name'], tried_name):
                            is_duplicate = True
                            print(f"[ToT] ⚠️ '{candidate['name']}' similar to '{tried_name}', skipping...")
                            stuck_count = 2  # Stay in alternative mode
                            continue
                    
                    if is_duplicate:
                        continue
                    
                    # Implement this candidate
                    alternatives_tried += 1
                    print(f"\n[ToT] Trying candidate #{alternatives_tried}/{max_alternatives}: {candidate['name']}")
                    print(f"[ToT] Score: {candidate['score']}/10")
                    print(f"[ToT] Reasoning: {candidate['reasoning'][:200]}...")
                    
                    # Save current best
                    best_solution = solution_files.copy()
                    best_score = test_results['passed']
                    print(f"[BACKUP] Saved current solution: {best_score}/{test_results['total']} passed")
                    
                    # Implement candidate
                    alternative = implement_architecture_tot(
                        problem_statement,
                        candidate,
                        solution_files
                    )
                    
                    if alternative:
                        arch_name = alternative.pop('__architecture_name__', candidate['name'])
                        solution_files.update(alternative)
                        write_files(alternative)
                        
                        # Give it iterations to improve
                        alt_iterations = max(3, min(4, MAX_ITERATIONS - iteration - 1))
                        print(f"[ToT] Running {alt_iterations} refinement iterations...")
                        
                        alt_stuck = 0
                        alt_prev_failed = set()
                        
                        for alt_iter in range(alt_iterations):
                            alt_success, alt_output = run_tests()
                            alt_results = parse_test_results(alt_output)
                            print(f"[ToT] Iteration {alt_iter+1}/{alt_iterations}: {alt_results['passed']}/{alt_results['total']} passed")
                            
                            if alt_success:
                                print(f"[ToT] ✓ All tests passed!")
                                break
                            
                            # Check if stuck
                            alt_curr_failed = set(alt_results['failed_tests'])
                            if alt_curr_failed == alt_prev_failed:
                                alt_stuck += 1
                                if alt_stuck >= 2:
                                    print(f"[ToT] Stuck on same {len(alt_curr_failed)} test(s), stopping early")
                                    break
                            else:
                                alt_stuck = 0
                            alt_prev_failed = alt_curr_failed
                            
                            # Fix failures
                            print(f"[ToT] Analyzing failures and generating fix...")
                            alt_fixed = fix_test_failures(problem_statement, alt_output, solution_files)
                            if alt_fixed:
                                solution_files.update(alt_fixed)
                                write_files(alt_fixed)
                            else:
                                break
                        
                        # Final test
                        final_success, final_output = run_tests()
                        final_results = parse_test_results(final_output)
                        
                        tried_architectures[arch_name] = final_results['passed']
                        
                        print(f"\n[COMPARISON] Original: {best_score}/{test_results['total']} | {arch_name}: {final_results['passed']}/{final_results['total']}")
                        
                        if final_results['passed'] > best_score:
                            print(f"[ToT] ✓ Better! Continuing with '{arch_name}'...")
                            stuck_count = 0
                            previous_failed = set(final_results['failed_tests'])
                            continue
                        else:
                            print(f"[ToT] ✗ Not better, reverting and trying next candidate...")
                            solution_files.update(best_solution)
                            write_files(best_solution)
                            stuck_count = 2
                            continue
                    else:
                        print(f"[ToT] Could not implement candidate")
                        stuck_count = 2
                        continue
        else:
            stuck_count = 0
        
        previous_failed = current_failed
        
        # Regular fix attempt
        print("[FIXING] Analyzing failures and generating fix...")
        fixed = fix_test_failures(problem_statement, output, solution_files)
        
        if not fixed:
            print("[WARNING] Could not generate fix")
            break
        
        write_files(fixed)
        solution_files.update(fixed)
    
    # Summary
    if tried_architectures:
        print(f"\n[SUMMARY] Tried {len(tried_architectures)} alternative architecture(s):")
        for arch_name, score in tried_architectures.items():
            print(f"  - {arch_name}: {score} tests passed")
    
    # Return final patch
    patch = get_git_diff()
    print(f"\n[COMPLETE] Generated patch ({len(patch)} bytes)")
    return patch

# ============================================================================
# Helper Functions
# ============================================================================

def detect_problem_type(problem_statement: str) -> str:
    """Detect if this is a CREATE or FIX problem."""
    fix_keywords = ["fix", "bug", "error", "issue", "broken", "failing", "incorrect"]
    statement_lower = problem_statement.lower()
    
    if any(keyword in statement_lower for keyword in fix_keywords):
        return "FIX"
    return "CREATE"

# ============================================================================
# Main Entry Point
# ============================================================================

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo") -> str:
    """
    Main entry point for the ToT agent.
    
    Args:
        input_dict: Dictionary with 'problem_statement' key
        repo_dir: Repository directory path
    
    Returns:
        Git diff patch as string
    """
    print("\n" + "="*80)
    print("TEST-DRIVEN AGENT (ToT)")
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
    timeout = DEFAULT_TIMEOUT - 120
    
    try:
        # Route to appropriate mode (only CREATE supported for now)
        if problem_type == "CREATE":
            patch = create_mode_tot(problem_statement, "tests.py")
        else:
            print(f"[WARNING] FIX mode not yet implemented in ToT, using CREATE mode")
            patch = create_mode_tot(problem_statement, "tests.py")
        
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

