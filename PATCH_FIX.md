# Fixed: Test Files in Patch

## The Problem

Even though the agent solved the problem perfectly, the patch couldn't be applied to the sandbox:

```
error: tests.py: already exists in working directory
```

## Root Cause Analysis

The original implementation had a mechanism to exclude test files from patches via `generated_test_files` list, but it wasn't being populated correctly.

**The Workflow:**
1. `process_create_task()` creates test files (`tests.py`) → Lines 4412-4418
2. Calls `fix_task_solve_workflow()` → Line 4440
3. Agent works and generates patch via `tool_manager.get_final_git_patch()` 
4. Patch generation reads `tool_manager.generated_test_files` to exclude files → Line 2608

**The Bug:**
- Test files were created in step 1 ✅
- But `test_files` list was **never passed** to `fix_task_solve_workflow` ❌
- So `tool_manager.generated_test_files` remained empty ❌
- So test files were included in the patch ❌

## The Fix

### 1. Added Parameter to fix_task_solve_workflow (Line 4480)

**Before:**
```python
def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str,
                            test_runner: str = "pytest", test_runner_mode: str = "FILE", 
                            n_max_steps=MAX_FIX_TASK_STEPS,
                            enable_pev: bool = True, enable_test_guidance: bool = True, 
                            extra_fix_request="") -> tuple[str, List[str], List[str]]:
```

**After:**
```python
def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str,
                            test_runner: str = "pytest", test_runner_mode: str = "FILE", 
                            n_max_steps=MAX_FIX_TASK_STEPS,
                            enable_pev: bool = True, enable_test_guidance: bool = True, 
                            extra_fix_request="",
                            generated_test_files: List[str] = None) -> tuple[str, List[str], List[str]]:
```

### 2. Populate tool_manager.generated_test_files (Lines 4546-4549)

**Added:**
```python
# Populate generated_test_files to exclude them from final patch
if generated_test_files:
    tool_manager.generated_test_files = generated_test_files[:]
    logger.info(f"[FIX_WORKFLOW] Tracking {len(generated_test_files)} test files to exclude from patch: {generated_test_files}")
```

### 3. Pass test_files When Calling fix_task_solve_workflow (Line 4450)

**Before:**
```python
patch = fix_task_solve_workflow(
    problem_statement,
    timeout=timeout,
    run_id_1=run_id,
    test_runner="unittest",
    test_runner_mode="FILE",
    n_max_steps=60,
    enable_pev=enable_pev,
    enable_test_guidance=enable_test_guidance,
    extra_fix_request=SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT
)
```

**After:**
```python
patch = fix_task_solve_workflow(
    problem_statement,
    timeout=timeout,
    run_id_1=run_id,
    test_runner="unittest",
    test_runner_mode="FILE",
    n_max_steps=60,
    enable_pev=enable_pev,
    enable_test_guidance=enable_test_guidance,
    extra_fix_request=SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT,
    generated_test_files=test_files  # ← NEW: Pass test files to exclude
)
```

## How It Works Now

```
Step 1: Create test files
        test_files = ['./tests.py']
        ↓
Step 2: Pass test_files to fix_task_solve_workflow(generated_test_files=test_files)
        ↓
Step 3: Populate tool_manager.generated_test_files = ['./tests.py']
        ↓
Step 4: Agent works...
        ↓
Step 5: Generate patch via get_final_git_patch()
        ↓
        exclude = {"src/agent.py", "src/agent_runner.py"}
        for _p in self.generated_test_files:  # ← ['./tests.py']
            exclude.add(os.path.relpath(_p))  # ← Adds 'tests.py' to exclude
        ↓
        to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
        # tests.py is now excluded! ✅
```

## Expected Log Output

You should now see:
```
[FIX_WORKFLOW] Tracking 1 test files to exclude from patch: ['./tests.py']
```

And the patch will only include solution files (main.py, etc.), not test files.

## Test It!

```bash
python ridges.py test-agent react sam_fix.py --timeout 50000
```

**Expected:**
- ✅ Agent solves in ~25 steps
- ✅ Patch generated
- ✅ Log shows: "Tracking 1 test files to exclude from patch: ['./tests.py']"
- ✅ Patch applies successfully to sandbox
- ✅ All tests pass in evaluation

## Why It Wasn't Working Before

The original implementation **was correct** - it had the `generated_test_files` mechanism. The bug was introduced when test file creation logic was fixed (lines 4412-4418) but the connection to pass `test_files` to the workflow was missed.

This is a perfect example of why tracking state across function boundaries is important! 🎯

## Summary

**Root Cause:** Test files not being tracked in `generated_test_files`  
**Fix:** Pass test files list through function parameters and populate tracking list  
**Result:** Test files now properly excluded from patches ✅

The agent's great work solving the problem will now actually be usable! 🚀
