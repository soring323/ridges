# Debug Mode Control

## Overview
All `[DEBUG]` logs in `test_driven_agent_oop.py` can now be controlled via the `DEBUG_MODE` environment variable.

## Usage

### Disable Debug Logs (Default)
```bash
# No environment variable needed - debug is OFF by default
python test_driven_agent_oop.py

# Or explicitly disable
export DEBUG_MODE=false
python test_driven_agent_oop.py
```

### Enable Debug Logs
```bash
# Enable debug mode
export DEBUG_MODE=true
python test_driven_agent_oop.py

# Or inline
DEBUG_MODE=true python test_driven_agent_oop.py
```

### For Local Benchmark
```bash
# Without debug (clean output)
python benchmark_agent.py

# With debug (verbose output)
DEBUG_MODE=true python benchmark_agent.py
```

### For Docker Sandbox
The validator can set this via environment variables when launching the sandbox:
```python
env_vars = {
    "DEBUG_MODE": "true"  # Enable debug logs in sandbox
}
```

## What Gets Disabled

When `DEBUG_MODE=false` (default), these logs are suppressed:
- ✅ `[DEBUG] Writing N files: ...`
- ✅ `[DEBUG] Successfully wrote: ...`
- ✅ `[DEBUG] Test output preview ...`
- ✅ `[DEBUG] Parsed: X passed, Y failed ...`
- ✅ `[DEBUG] Received N candidates from parallel generation`
- ✅ `[DEBUG] Candidate is_perfect=...`
- ✅ `[DEBUG] Perfect candidate added to list!`
- ✅ `[DEBUG] Checking for perfect solutions: ...`
- ✅ `[DEBUG] Perfect candidate: ...`
- ✅ `[DEBUG] After sort, candidates[0]: ...`
- ✅ `[DEBUG] Added dynamic future to tracking ...`

## What Still Shows

Important logs that always show (regardless of DEBUG_MODE):
- `[AGENT]` - Agent initialization and status
- `[SETUP]` - Setup and configuration
- `[STEP N]` - Step-by-step progress
- `[ROUND N]` - Round progress
- `[PARALLEL]` - Parallel execution status
- `[COT]` - Chain of thought
- `[NETWORK]` - Network errors
- `[ERROR]` - All errors
- `[SUCCESS]` - Success messages
- `[COMPLETE]` - Completion status

## Implementation

The debug system uses a simple helper function:
```python
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

def debug_print(*args, **kwargs):
    """Print only if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)
```

All debug logs use `debug_print()` instead of `print()`.

## Example Output Comparison

### Without Debug (Clean)
```
[AGENT] Test-Driven Agent initialized - Mode: CREATE
[SETUP] Working directory: /sandbox/repo
[STEP 1] Checking for existing test files...
[ROUND 1] Best candidate: 'State machine...' (8/8 tests)
[SUCCESS] Perfect solution found in attempt 1!
```

### With Debug (Verbose)
```
[AGENT] Test-Driven Agent initialized - Mode: CREATE
[SETUP] Working directory: /sandbox/repo
[DEBUG] Writing 1 files: ['main.py']
[DEBUG] Successfully wrote: ['main.py']
[STEP 1] Checking for existing test files...
[DEBUG] Test output preview (1185 chars total):
===== test session starts =====
[DEBUG] Parsed: 8 passed, 0 failed, 8 total
[DEBUG] Received 6 candidates from parallel generation
[DEBUG]   1. State machine... - 8/8 (is_perfect=True)
[ROUND 1] Best candidate: 'State machine...' (8/8 tests)
[DEBUG] Best candidate is_perfect: True
[DEBUG] ✅ Perfect candidate added to list! Total candidates: 1
[SUCCESS] Perfect solution found in attempt 1!
```

## Recommendation

**For production/evaluation**: Keep `DEBUG_MODE=false` (default) for clean, readable logs
**For development/debugging**: Set `DEBUG_MODE=true` to see detailed execution flow
