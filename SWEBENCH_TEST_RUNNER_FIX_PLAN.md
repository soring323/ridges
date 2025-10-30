# SWE-bench Test Runner Integration - Fix Plan

## Problem
Agent has no knowledge of SWE-bench test specifications (fail_to_pass, pass_to_pass) and uses wrong test discovery method.

## Solution Options

### Option 1: Pass Test Specs to Agent (RECOMMENDED)

Modify agent input to include test specifications:

```python
# In validator when calling agent:
input_dict = {
    "problem_statement": problem.get("problem_statement"),
    "test_specs": {
        "fail_to_pass": problem.get("FAIL_TO_PASS"),  
        "pass_to_pass": problem.get("PASS_TO_PASS")
    }
}
```

Then in agent (sam.py):

```python
# process_fix_task()
test_specs = input_dict.get("test_specs", {})

# Pass to workflow
patch_text = fix_task_solve_workflow(
    problem_text,
    test_specs=test_specs,  # NEW
    ...
)

# In tool_manager.run_repo_tests():
if self.test_specs:
    # Run using TEST_RUNNER.py format
    fail_to_pass = self.test_specs.get("fail_to_pass", [])
    pass_to_pass = self.test_specs.get("pass_to_pass", [])
    cmd = f"pytest {' '.join(fail_to_pass + pass_to_pass)}"
else:
    # Fallback to glob patterns
    ...
```

### Option 2: Use TEST_RUNNER.py Directly

Import and use the validator's TEST_RUNNER.py in the agent:

```python
# In tool_manager
from validator.problem_suites.swebench_verified.TEST_RUNNER import run_tests

def run_repo_tests(self, file_paths=None):
    if self.test_specs:
        results = run_tests(self.test_specs)
        return format_results(results)
    else:
        # Fallback...
```

### Option 3: Environment Variable

Set tests as environment variable before agent runs:

```python
# In validator before running agent
os.environ["SWEBENCH_TESTS"] = json.dumps({
    "fail_to_pass": [...],
    "pass_to_pass": [...]
})

# In agent
test_specs = json.loads(os.getenv("SWEBENCH_TESTS", "{}"))
```

## Recommended Implementation (Option 1)

### Step 1: Modify agent input interface

File: `validator/problem_suites/swebench_verified/swebench_verified_suite.py`

```python
def run_agent(self, problem):
    input_dict = {
        "problem_statement": problem.get("problem_statement"),
        "instance_id": problem.get("instance_id"),
        "test_specs": problem.get("tests")  # Add this
    }
    agent_result = agent_main(input_dict)
    return agent_result
```

### Step 2: Update agent to receive tests

File: `sam.py` - process_fix_task()

```python
def process_fix_task(input_dict: Dict[str, Any], ...):
    problem_text = input_dict.get("problem_statement")
    test_specs = input_dict.get("test_specs", {})  # NEW
    
    patch_text = fix_task_solve_workflow(
        problem_text,
        test_specs=test_specs,  # NEW
        ...
    )
```

### Step 3: Update workflow signature

File: `sam.py` - fix_task_solve_workflow()

```python
def fix_task_solve_workflow(
    problem_statement: str, 
    test_specs: dict = None,  # NEW
    *, 
    timeout: int, 
    ...
):
    ...
    tool_manager = FixTaskEnhancedToolManager(
        ...
        test_specs=test_specs  # NEW
    )
```

### Step 4: Update tool manager

File: `sam.py` - FixTaskEnhancedToolManager.__init__()

```python
def __init__(self, ..., test_specs: dict = None):
    ...
    self.test_specs = test_specs or {}
```

### Step 5: Fix run_repo_tests tool

File: `sam.py` - run_repo_tests()

```python
def run_repo_tests(self, file_paths=None):
    # If we have SWE-bench test specs, use them
    if self.test_specs:
        fail_to_pass = self.test_specs.get("fail_to_pass", [])
        pass_to_pass = self.test_specs.get("pass_to_pass", [])
        
        if fail_to_pass or pass_to_pass:
            all_tests = fail_to_pass + pass_to_pass
            logger.info(f"[RUN_TESTS] Running SWE-bench tests: {len(all_tests)} total")
            
            # Run pytest with specific test paths
            cmd = f"pytest -xvs {' '.join(all_tests)}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
            return result.stdout + result.stderr
    
    # Fallback to original glob-based discovery
    if file_paths is None:
        file_paths = glob.glob("test*.py") + glob.glob("*test.py")
        file_paths = [f for f in file_paths if f != "conftest.py"]  # Exclude conftest
    ...
```

### Step 6: Fix finish tool test discovery

File: `sam.py` - finish()

```python
def finish(self, investigation_summary: str):
    ...
    # Use SWE-bench tests if available
    if self.test_specs:
        fail_to_pass = self.test_specs.get("fail_to_pass", [])
        pass_to_pass = self.test_specs.get("pass_to_pass", [])
        test_output = self.run_repo_tests()  # Will use test_specs internally
    else:
        # Fallback to file-based discovery
        all_test_files = glob.glob("test*.py") + glob.glob("*test.py")
        all_test_files = [f for f in all_test_files if f != "conftest.py"]
        ...
```

## Testing Plan

1. Add test_specs to a sample problem input
2. Verify agent receives and uses them
3. Check that run_repo_tests executes correct tests
4. Verify finish tool validates correct tests
5. Confirm validator's TEST_RUNNER.py matches results

## Files to Modify

1. `validator/problem_suites/swebench_verified/swebench_verified_suite.py` - pass tests to agent
2. `sam.py`:
   - `process_fix_task()` - receive test_specs
   - `fix_task_solve_workflow()` - pass test_specs to tool_manager
   - `FixTaskEnhancedToolManager.__init__()` - store test_specs
   - `run_repo_tests()` - use test_specs if available
   - `finish()` - use test_specs for validation

## Expected Outcome

After fixes:
- Agent knows exactly which tests to run
- No more conftest.py false positives
- run_repo_tests matches validator's TEST_RUNNER.py
- finish tool validates correct tests
- Proper integration between agent and validator
