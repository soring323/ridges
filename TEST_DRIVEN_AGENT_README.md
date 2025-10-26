# Test-Driven Agent - Architecture & Flow Documentation

## Overview
This is a test-driven development agent that solves coding problems by generating solutions, running tests, and iteratively refining until all tests pass. It supports both **CREATE mode** (build from scratch) and **FIX mode** (debug existing code), with optional parallel solution generation.

**Total Lines:** 2,088 lines
**Complexity:** HIGH - Multiple managers, parallel execution, multiple refinement strategies

---

## 🎯 Main Entry Point

### `agent_main(input_data: Dict) -> str` (Lines 1933-2057)

**Purpose:** Main entry point called by benchmark system

**Flow:**
```
1. Parse input (problem_statement, timeout, mode)
2. Setup git repository for patch generation
3. Auto-detect mode (CREATE or FIX) if not specified
4. Initialize concrete implementations:
   - PytestRunner (test execution)
   - LLMCodeGenerator (code generation)
   - LLMArchitectureGenerator (alternative architectures)
   - LocalFileManager (file I/O)
5. Create TestDrivenAgent with configuration
6. Call agent.solve_create() or agent.solve_fix()
7. Return git diff patch
```

---

## 📋 Execution Flow - CREATE Mode

### `TestDrivenAgent.solve_create()` (Lines 1427-1580)

**Main CREATE mode flow with parallel option:**

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Test Generation                                     │
├─────────────────────────────────────────────────────────────┤
│ • Check for existing test files (glob *test*.py)            │
│ • If none found: LLMCodeGenerator.generate_tests()          │
│   - Uses CODING_MODEL                                        │
│   - 2-step validation (generate + validate)                 │
│ • Write test files to disk                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Solution Generation (PARALLEL or SEQUENTIAL)        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ IF ENABLE_PARALLEL == TRUE:                                 │
│   → ParallelSolutionGenerator.generate_multiple_solutions()│
│   → Loop: 4 rounds × num_workers solutions                  │
│   → Each round:                                              │
│       1. Generate diverse architectures (COT reasoning)     │
│       2. Generate N solutions in parallel                    │
│       3. Pick best candidate                                 │
│       4. Refine best with RefinementLoop                    │
│       5. Break if perfect solution found                     │
│                                                              │
│ ELSE (Sequential):                                           │
│   → LLMCodeGenerator.generate_solution()                    │
│   → Single initial solution                                  │
│   → Run baseline tests                                       │
│   → RefinementLoop for iterative fixes                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Generate Git Diff Patch                             │
├─────────────────────────────────────────────────────────────┤
│ • get_git_diff_helper()                                      │
│ • Returns unified diff format                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Execution Flow - FIX Mode

### `TestDrivenAgent.solve_fix()` (Lines 1582-1619)

**Simpler flow for debugging:**

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Find Relevant Files                                 │
├─────────────────────────────────────────────────────────────┤
│ • LocalFileManager.read_files(['*.py'])                     │
│ • Loads all Python files in workspace                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Generate Reproduction Test                          │
├─────────────────────────────────────────────────────────────┤
│ • LLMCodeGenerator.generate_tests(problem, relevant_files) │
│ • Creates test that reproduces the bug                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Iterative Fixing                                    │
├─────────────────────────────────────────────────────────────┤
│ • RefinementLoop.run()                                      │
│ • Fewer alternatives than CREATE mode (5 vs 10)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Generate Patch                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Core Component: RefinementLoop

### `RefinementLoop.run()` (Lines 1337-1403)

**Iterative test-fix cycle:**

```python
For each iteration (up to max_iterations=20):
    1. TestManager.run_and_parse()
       ├─ PytestRunner.run_tests()
       └─ PytestRunner.parse_results()
    
    2. Check if all tests pass → SUCCESS, return
    
    3. Track progress:
       ├─ If score improved: reset stuck counter
       └─ If no improvement for 3 iterations: warn
    
    4. Check if stuck (same tests failing):
       ├─ If stuck for stuck_threshold (2) iterations: BREAK
       └─ Else: reset stuck counter
    
    5. FixManager.apply_fix()
       ├─ LLMCodeGenerator.fix_failures()
       │  ├─ Extract detailed failure info
       │  ├─ Call LLM with failure context
       │  └─ Parse fixed code blocks
       └─ Write fixed files to disk
    
    6. Check timeout
```

**Key Variables:**
- `best_score`: Tracks highest test pass count
- `stuck_count`: Counts consecutive iterations with same failures
- `iterations_without_improvement`: Counts iterations without score increase

---

## 🚀 Parallel Solution Generation

### `ParallelSolutionGenerator.generate_multiple_solutions()` (Lines 1850-1927)

**Multi-threaded solution generation with architecture diversity:**

```
1. Generate N diverse architectures using COT:
   └─ generate_diverse_architectures()
      ├─ Build prompt with:
      │  • Problem summary
      │  • List of previously tried architectures (avoid duplicates)
      │  • Failure analysis from best previous attempt (if available)
      ├─ Call LLM with high temperature (0.9) for creativity
      └─ Parse N architecture descriptions
      
2. Parallel execution:
   └─ ThreadPoolExecutor with num_workers threads
      ├─ For each architecture hint:
      │  └─ Submit generate_and_test_solution()
      │     ├─ Generate solution with architecture hint
      │     ├─ Write files (thread-safe with lock)
      │     ├─ Run tests (thread-safe with lock)
      │     └─ Return SolutionCandidate
      │
      └─ Early termination if perfect solution found
      
3. Sort candidates by score (best first)

4. Return all candidates
```

**Thread Safety:**
- `file_lock`: Protects file I/O operations
- `termination_lock`: Protects `perfect_solution_found` flag
- `architecture_lock`: Protects `used_architecture_descriptions` list

---

## 🏗️ Architecture Classes & Responsibilities

### Core Interfaces (Lines 143-201)
```
ITestRunner         → Run tests, parse results
ICodeGenerator      → Generate solutions, tests, fixes
IArchitectureGenerator → Generate alternative architectures
IFileManager        → Write/read files
```

### Concrete Implementations (Lines 510-1067)
```
PytestRunner               → Executes pytest, parses output
LLMCodeGenerator           → Calls LLM API for code generation
LLMArchitectureGenerator   → Generates alternative architectures
LocalFileManager           → File I/O operations
```

### Managers (Lines 1088-1316)
```
TestManager          → Wraps test runner, formats output
FixManager           → Applies fixes iteratively
ArchitectureManager  → Manages alternative architecture exploration
```

### Main Orchestrator (Lines 1409-1623)
```
TestDrivenAgent → Facade that coordinates all components
```

---

## 🔍 Detailed Function Call Order - CREATE Mode (Parallel)

```
agent_main()
  └─ TestDrivenAgent.solve_create()
      │
      ├─ [STEP 1: Tests]
      │   ├─ Path('.').glob('*test*.py')  # Check existing tests
      │   ├─ LLMCodeGenerator.generate_tests() [if no tests]
      │   │   ├─ call_llm(CODING_MODEL) [Step 1: Generate]
      │   │   ├─ call_llm(CODING_MODEL) [Step 2: Validate]
      │   │   └─ parse_file_blocks()
      │   └─ LocalFileManager.write_files()
      │
      ├─ [STEP 2: Parallel Generation]
      │   └─ ParallelSolutionGenerator
      │       │
      │       ├─ FOR round 1 to 4:
      │       │   │
      │       │   ├─ generate_diverse_architectures()
      │       │   │   ├─ call_llm(REASONING_MODEL, temp=0.9)
      │       │   │   └─ Parse architecture descriptions
      │       │   │
      │       │   ├─ generate_multiple_solutions()
      │       │   │   └─ ThreadPoolExecutor:
      │       │   │       └─ For each architecture:
      │       │   │           └─ generate_and_test_solution()
      │       │   │               ├─ LLMCodeGenerator.generate_solution(hint)
      │       │   │               │   ├─ call_llm(REASONING_MODEL)
      │       │   │               │   └─ parse_file_blocks()
      │       │   │               ├─ [LOCK] write_files()
      │       │   │               ├─ [LOCK] TestManager.run_and_parse()
      │       │   │               │   ├─ PytestRunner.run_tests()
      │       │   │               │   │   └─ subprocess.run(pytest)
      │       │   │               │   └─ PytestRunner.parse_results()
      │       │   │               │       └─ parse_test_results_helper()
      │       │   │               └─ Return SolutionCandidate
      │       │   │
      │       │   ├─ Pick best candidate (highest score)
      │       │   │
      │       │   ├─ IF perfect: BREAK
      │       │   │
      │       │   └─ ELSE: RefinementLoop.run()
      │       │       └─ FOR iteration 1 to max_iterations:
      │       │           ├─ TestManager.run_and_parse()
      │       │           ├─ Check success → return
      │       │           ├─ Track progress/stuck
      │       │           └─ FixManager.apply_fix()
      │       │               ├─ LLMCodeGenerator.fix_failures()
      │       │               │   ├─ _extract_failure_summary()
      │       │               │   ├─ call_llm(CODING_MODEL)
      │       │               │   └─ parse_file_blocks()
      │       │               └─ write_files()
      │       │
      │       └─ Return best overall solution
      │
      └─ [STEP 3: Patch]
          └─ get_git_diff_helper()
              └─ subprocess.run(git diff)
```

---

## ⚠️ IDENTIFIED ISSUES & INEFFICIENCIES

### 🔴 CRITICAL ISSUES

#### 1. **Unused Code - ArchitectureManager.try_alternative() (Lines 1256-1309)**
- **Issue:** This entire method is marked as "## Unused" and is never called
- **Impact:** Dead code, 54 lines that confuse the codebase
- **Recommendation:** DELETE this method

#### 2. **Redundant Architecture Similarity Check**
- **Issue:** `are_architectures_similar_helper()` (Lines 1073-1082) is only used by `ArchitectureManager._are_similar()`
- **Current Usage:** Only in unused `try_alternative()` method
- **Recommendation:** DELETE if `try_alternative()` is removed

#### 3. **Double Test Generation in generate_tests()**
- **Issue:** Tests are generated twice with 2 LLM calls (Lines 664-718)
  - First call: Generate tests
  - Second call: Validate tests (often returns same result)
- **Cost:** 2× API cost, 2× latency
- **Recommendation:** Make validation optional or only validate if first generation seems problematic

#### 4. **Expensive Lock Contention in Parallel Mode**
- **Issue:** `file_lock` serializes ALL file writes and test runs (Lines 1817-1822)
- **Impact:** Threads wait for each other, defeating parallelism benefit
- **Current:** N threads but effectively sequential for testing
- **Recommendation:** 
  - Use temporary directories per thread
  - Only lock when selecting best candidate

#### 5. **Architecture Generation Not Used in Sequential Mode**
- **Issue:** `LLMArchitectureGenerator` only used in parallel mode
- **Impact:** Sequential mode stuck with single architecture, no alternatives
- **Recommendation:** Use alternative architectures in sequential mode too

---

### 🟡 MODERATE ISSUES

#### 6. **Inefficient Context Truncation**
- **Issue:** Multiple truncation functions doing similar things:
  - `truncate_text()` (Lines 206-217)
  - `truncate_messages()` (Lines 223-245)
  - `_extract_failure_summary()` (Lines 740-769)
- **Impact:** Duplicate logic, inconsistent truncation
- **Recommendation:** Unified truncation strategy

#### 7. **Redundant Test Parsing**
- **Issue:** Test results parsed twice:
  - Once in `run_tests_helper()` to check success
  - Again in `parse_test_results_helper()` for structured data
- **Recommendation:** Single parsing pass

#### 8. **RefinementLoop Doesn't Use ArchitectureManager**
- **Issue:** `RefinementLoop` has an `arch_manager` but never calls it
- **Location:** Lines 1322-1403, `arch_manager` only used for `print_summary()`
- **Impact:** Alternative architectures not tried during refinement
- **Recommendation:** Either use it or remove it from constructor

#### 9. **Stuck Detection Duplicated**
- **Issue:** Stuck detection logic appears in:
  - `FixManager.refine_iteratively()` (Lines 1208-1216)
  - `RefinementLoop.run()` (Lines 1380-1391)
- **Recommendation:** Extract to shared helper

#### 10. **Large Prompts with Repeated Context**
- **Issue:** 
  - `fix_failures()` prompt: 300+ lines (Lines 808-905)
  - `generate_alternative()` prompt: 200+ lines (Lines 937-1015)
  - Both contain similar instructions
- **Impact:** Token waste, slower inference
- **Recommendation:** Extract common instructions to shared template

---

### 🟢 MINOR ISSUES

#### 11. **Confusing Variable Names**
- `files_summary` vs `solution_summary` - inconsistent naming
- `prev_failed` vs `previous_failed` vs `current_failed`
- Recommendation: Standardize naming

#### 12. **Magic Numbers**
- Line 1467: `max_rounds = 4` - should be config
- Line 1469: `solutions_per_round = parallel_gen.num_workers` - why?
- Line 134: `stuck_threshold: int = 2` - not explained
- Recommendation: Add constants with explanatory comments

#### 13. **Excessive Debug Printing**
- 50+ print statements throughout
- No log levels (DEBUG, INFO, WARNING)
- Recommendation: Use proper logging module

#### 14. **Missing Error Handling**
- `generate_and_test_solution()` catches all exceptions (Line 1841)
- Returns empty candidate instead of logging specific error
- Makes debugging difficult

#### 15. **Test Results Success Logic Inconsistency**
- Line 122: `success = passed > 0 and failed == 0`
- Line 372: `success = returncode == 0`
- Different definitions of success

---

## 📊 Complexity Metrics

### Classes & Responsibilities
- **8 Data Classes** (TestResults, SolutionCandidate, etc.)
- **4 Abstract Interfaces** (ITestRunner, ICodeGenerator, etc.)
- **4 Concrete Implementations**
- **4 Manager Classes** (TestManager, FixManager, etc.)
- **2 Orchestrators** (TestDrivenAgent, ParallelSolutionGenerator)

### Function Count by Category
- **Helper Functions:** 9 (truncate, parse, run_tests, etc.)
- **LLM Code Generation:** 3 (generate_solution, generate_tests, fix_failures)
- **Test Execution:** 3 (run_tests, parse_results, print_summary)
- **File Management:** 3 (write_files, read_files, get_git_diff)
- **Refinement:** 3 (refine_iteratively, run, apply_fix)
- **Parallel Generation:** 3 (generate_multiple_solutions, generate_and_test_solution, generate_diverse_architectures)

### External Dependencies
- **subprocess:** Git operations, pytest execution
- **requests:** LLM API calls
- **psutil:** CPU usage monitoring
- **ThreadPoolExecutor:** Parallel execution
- **pytest:** Test framework (runtime dependency)

---

## 🎯 Optimization Recommendations

### High Priority
1. **Remove unused code:** `try_alternative()` and `are_architectures_similar_helper()`
2. **Fix lock contention:** Use per-thread temp directories
3. **Single-pass test validation:** Remove double generation in `generate_tests()`
4. **Use ArchitectureManager in RefinementLoop:** Or remove from constructor

### Medium Priority
5. **Unified truncation strategy:** Single function with clear token budgets
6. **Extract common prompt templates:** Reduce token usage
7. **Add proper logging:** Replace print statements
8. **Configuration constants:** Move magic numbers to config

### Low Priority
9. **Standardize variable naming:** Consistent conventions
10. **Better error messages:** Log specific errors in parallel mode
11. **Test success definition:** Align definitions across codebase

---

## 📝 Recommended Flow Simplification

### Current Parallel Flow (Complex)
```
4 rounds × N workers × (generate + test + lock)
→ Pick best
→ Refine best (max_iterations)
→ If stuck, no alternatives tried
```

### Suggested Flow (Simpler & More Effective)
```
Round 1: Generate N diverse solutions in parallel (no lock contention)
→ Quickly score all N solutions in parallel
→ Pick top 3 candidates
→ Refine each candidate in parallel
→ If any stuck, try alternative architecture
→ Return best overall
```

**Benefits:**
- Less sequential bottleneck
- Alternative architectures used when stuck
- Better CPU utilization

---

## 🔑 Key Takeaways

### Strengths
✅ Well-structured OOP design with clear interfaces  
✅ Parallel solution generation for faster exploration  
✅ Comprehensive test-driven refinement loop  
✅ Detailed failure analysis for LLM context  
✅ Support for both CREATE and FIX modes  

### Weaknesses
❌ Lock contention negates parallelism benefits  
❌ Unused code and dead branches  
❌ Double test generation wastes API calls  
❌ Alternative architectures only in parallel mode  
❌ No structured logging  
❌ Inconsistent success definitions  

### Overall Assessment
**Complexity Score: 8/10** (Very Complex)  
**Maintainability: 6/10** (Good structure, but too many layers)  
**Performance: 7/10** (Parallel execution hampered by locks)  
**Code Quality: 7/10** (Clean OOP, but unused code and inconsistencies)

---

## 📚 Additional Notes

### Model Selection Strategy
- **REASONING_MODEL:** DeepSeek-V3 (for architecture design, CoT)
- **CODING_MODEL:** Qwen3-Coder-480B (for code generation/fixing)
- **FAST_MODEL:** DeepSeek-V3 (not used in current code)

### Temperature Settings
- **0.0:** Deterministic (fix_failures, validation)
- **0.8:** Creative (alternative architectures)
- **0.9:** Maximum diversity (architecture generation)

### Resource Management
- **Optimal workers:** 2-8 threads based on CPU load
- **Dynamic scaling:** Adjusts based on `psutil.cpu_percent()`
- **Min 2, Max 8:** Bounds to prevent overload

---

**Document Version:** 1.0  
**Created:** Based on analysis of test_driven_agent_oop_update.py (2,088 lines)  
**Recommendation:** Simplify by removing unused code and fixing lock contention issue
