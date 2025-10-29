# CRITICAL BUG: Workflow Exits Immediately After Init

## Evidence from Logs

```
2025-10-29 12:29:01,705 - INFO - [FIX_WORKFLOW] ENTERED fix_task_solve_workflow
2025-10-29 12:29:01,705 - INFO - [FIX_WORKFLOW] timeout=2000s, n_max_steps=400
2025-10-29 12:29:01,705 - INFO - [FIX_WORKFLOW] ✓ PEVWorkflow initialized
HEAD is now at 91f2e47 Initial commit  ← PROCESS EXITS HERE!
[AGENT_RUNNER] Exited agent's agent_main()
```

## The Mystery

**Expected:**
1. Enter workflow ✅
2. Initialize PEVWorkflow ✅
3. Run strategy planning ❌
4. Initialize guidance ❌
5. Enter main loop (Step 1, 2, 3...) ❌
6. Generate patch ❌
7. Exit normally ❌

**Actual:**
1. Enter workflow ✅
2. Initialize PEVWorkflow ✅
3. **IMMEDIATELY EXITS** - no further logs ❌

## Critical Missing Logs

These logs were added but NEVER appeared:
- `[FIX_WORKFLOW] Problem type determined: bug_fix`
- `[FIX_WORKFLOW] Starting strategy planning phase...`
- `[FIX_WORKFLOW] ✓ Strategy selected: X`
- `[FIX_WORKFLOW] === Step 1/400 ===`
- `[CREATE_TASK] fix_task_solve_workflow completed`

**This means the process crashed/exited between lines 4433 and 4442 of sam_fix.py**

## Root Cause Hypothesis

### Theory 1: Silent Exception (**Most Likely**)
Something raises an exception after PEVWorkflow init, but it's being caught by an outer try-catch that doesn't log it.

**Evidence:**
- Logs stop abruptly
- Git reset happens (cleanup from test harness)
- Process exits cleanly (no crash dump)

### Theory 2: Process Timeout
The agent process might have a timeout that kills it.

**Evidence Against:**
- timeout=2000s (plenty of time)
- Only ~2 seconds elapsed at init point
- No timeout warning in logs

### Theory 3: sys.exit() Call
Hidden sys.exit() or os._exit() somewhere.

**Evidence Against:**
- Would need to be between line 4433 and 4442
- No such code visible in that section

### Theory 4: Signal Handler
External process killing the Python agent.

**Evidence:**
- Git reset happens immediately
- Could be test harness detecting a problem

## What We Added to Catch It

### 1. Exception Handling Around PEVWorkflow Init
```python
try:
    pev = PEVWorkflow(...)
    logger.info("[FIX_WORKFLOW] ✓ PEVWorkflow initialized")
except Exception as e:
    logger.error(f"[FIX_WORKFLOW] FATAL: PEVWorkflow initialization failed: {e}")
    logger.error(f"[FIX_WORKFLOW] Traceback: {traceback.format_exc()}")
    raise
```

### 2. Exception Handling Around Strategy Planning
```python
logger.info("[FIX_WORKFLOW] Starting strategy planning phase...")
try:
    strategy = pev.run_planning_phase(...)
    logger.info(f"[FIX_WORKFLOW] ✓ Strategy selected: {strategy.get('name', 'Unknown')}")
except Exception as e:
    logger.error(f"[FIX_WORKFLOW] FATAL: Strategy planning failed: {e}")
    logger.error(f"[FIX_WORKFLOW] Traceback: {traceback.format_exc()}")
    raise
```

### 3. Exception Handling Around Entire Workflow Call
```python
patch = None
try:
    patch = fix_task_solve_workflow(...)
    logger.info("[CREATE_TASK] fix_task_solve_workflow completed SUCCESSFULLY")
except Exception as e:
    logger.error(f"[CREATE_TASK] FATAL: fix_task_solve_workflow CRASHED!")
    logger.error(f"[CREATE_TASK] Exception: {e}")
    logger.error(f"[CREATE_TASK] Traceback:\n{traceback.format_exc()}")
    raise
```

### 4. Progress Logging
```python
logger.info(f"[FIX_WORKFLOW] Problem type determined: {problem_type}")
logger.info("[FIX_WORKFLOW] Starting strategy planning phase...")
```

## Next Run - What to Look For

### Scenario A: Exception Caught
```
[FIX_WORKFLOW] ✓ PEVWorkflow initialized
[FIX_WORKFLOW] Problem type determined: bug_fix
[FIX_WORKFLOW] Starting strategy planning phase...
[FIX_WORKFLOW] FATAL: Strategy planning failed: KeyError('strategies')
[FIX_WORKFLOW] Traceback: ...
[CREATE_TASK] FATAL: fix_task_solve_workflow CRASHED!
```

**Action:** Fix the specific exception

### Scenario B: External Timeout/Kill
```
[FIX_WORKFLOW] ✓ PEVWorkflow initialized
[FIX_WORKFLOW] Problem type determined: bug_fix
[FIX_WORKFLOW] Starting strategy planning phase...
(no more logs - process killed externally)
HEAD is now at...
```

**Action:** Investigate test harness timeout settings

### Scenario C: Still Exits at Same Point
```
[FIX_WORKFLOW] ✓ PEVWorkflow initialized
HEAD is now at...
(no additional logs)
```

**Action:** 
- Exception is being raised INSIDE PEVWorkflow.__init__() but after our try-catch
- Need to add logging inside PEVWorkflow itself
- Check if there's a __del__() destructor causing issues

## Suspicious Code Locations

### PEVWorkflow.__init__()
```python
def __init__(self, enable_pev: bool = True, enable_test_guidance: bool = True, ...):
    self.enable_pev = enable_pev
    # ... more init ...
    if enable_pev:
        self.planner = StrategicPlanner()  ← Could crash here
        self.verifier = Verifier()          ← Or here
        if enable_test_guidance:
            self.guidance = TestDrivenGuidance()  ← Or here
```

**If StrategicPlanner(), Verifier(), or TestDrivenGuidance() raise exceptions, they would happen INSIDE PEVWorkflow init.**

### StrategicPlanner.__init__()
```python
def __init__(self, model_name: str = DEEPSEEK_MODEL_NAME):
    self.model_name = model_name
    self.outcome_tracker = StrategyOutcomeTracker()  ← Could crash loading file
    self.selected_strategy = None
```

### StrategyOutcomeTracker.__init__()
```python
def __init__(self, outcomes_file: str = ".strategy_outcomes.json"):
    self.outcomes_file = outcomes_file
    self.outcomes = self._load_outcomes()  ← Could crash loading JSON
```

### TestDrivenGuidance.__init__()
```python
def __init__(self, trace_learner: TraceBasedLearning = None):
    self.trace_learner = trace_learner or TraceBasedLearning()  ← Could crash
    self.api_analyzer = APIPatternAnalyzer()
```

### TraceBasedLearning.__init__()
```python
def __init__(self, trace_file: str = ".agent_traces.json"):
    self.trace_file = trace_file
    self.traces = self._load_traces()  ← Could crash loading JSON
```

## Most Likely Culprit

**StrategyOutcomeTracker or TraceBasedLearning are trying to load corrupted JSON files!**

The files `.strategy_outcomes.json` or `.agent_traces.json` might:
- Not exist (should be handled)
- Be corrupted JSON (should be handled by try-catch, but might not be)
- Have permission issues
- Be locked by another process

## Immediate Fix If JSON Loading is the Problem

Both classes have this pattern:
```python
try:
    with open(self.trace_file, 'r') as f:
        return json.load(f)
except (FileNotFoundError, json.JSONDecodeError, IOError):
    return []
```

But what if there's a **different exception**? Like:
- PermissionError
- OSError (disk full)
- UnicodeDecodeError
- IsADirectoryError

**Solution:** Change to catch all exceptions:
```python
except Exception as e:
    logger.warning(f"Failed to load {self.trace_file}: {e}")
    return []
```

---

## Summary

**The agent workflow is crashing somewhere in the initialization chain:**
```
fix_task_solve_workflow() 
  → PEVWorkflow() 
    → StrategicPlanner() 
      → StrategyOutcomeTracker() 
        → _load_outcomes()  ← CRASH HERE?
```

**With the new logging, the next run will show us EXACTLY where and why!**

Run again and check for:
1. ✅ Exception messages with tracebacks
2. ✅ Which initialization step fails
3. ✅ Specific error that's killing the process

The mystery will be solved! 🔍
