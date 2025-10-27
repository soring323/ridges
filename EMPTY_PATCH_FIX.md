# Empty Patch Fix (0 bytes)

## Problem
Evaluation failing with:
```
error: No valid patches in input (allow with "--allow-empty")
[COMPLETE] Generated patch (0 bytes)
```

## Root Causes

### 1. Git Ownership Issue
```
fatal: detected dubious ownership in repository at '/sandbox/repo'
```
Docker sandbox runs as different user → git refuses operations → git commands fail silently

### 2. Wrong Filename
- Agent created `beer_song.py` (detected from generated tests)
- Git repo only tracks `main.py` (from polyglot dataset)
- `git diff` shows nothing → 0 byte patch

## Solutions

### Fix 1: Git Ownership
Added at the start of `agent_main()`:
```python
# Fix git ownership issue in Docker sandbox
subprocess.run(
    ["git", "config", "--global", "--add", "safe.directory", os.getcwd()],
    capture_output=True
)
```

This allows git operations in Docker sandbox where directory is owned by different user.

### Fix 2: Always Write to main.py
Changed in `LLMCodeGenerator.generate_solution()` and `fix_failures()`:
```python
# CRITICAL: Always use main.py for the patch (it's the only file tracked in git)
if "main.py" not in files:
    first_key = list(files.keys())[0]
    content = files[first_key]
    files = {"main.py": content}
    print(f"[CODE_GEN] ⚠️ Renamed '{first_key}' to 'main.py' for git patch")
```

## Why This Works

### Agent Sandbox (Development):
1. Starts with `main.py` (skeleton) tracked in git
2. Agent detects module name from its generated tests (might be `beer_song`)
3. Agent writes code → **forced to `main.py`**
4. `git diff main.py` → valid patch

### Evaluation Sandbox:
1. Starts with `main.py` (skeleton) + `tests.py` (from dataset)
2. Tests import from `main` (as per dataset, e.g., `from main import recite`)
3. Patch applied to `main.py`
4. Tests run successfully

## Key Insight

The agent generates its own tests during development (which might import from any module), but **evaluation uses the original tests.py from the dataset** (which imports from `main`).

Therefore:
- ✅ Agent must write to `main.py` (for git patch)
- ✅ Evaluation tests import from `main` (from dataset)
- ✅ Everything aligns!

## Testing

Run with debug to see the fix:
```bash
DEBUG_MODE=true python ridges.py test-agent beer-song test_driven_agent_oop.py
```

Look for:
```
[CODE_GEN] ⚠️ Renamed 'beer_song.py' to 'main.py' for git patch (tests import from 'beer_song')
[GIT] Git status before staging:
 M main.py
[GIT] Generated patch (5646 bytes)  ← Should NOT be 0!
```

## Result

✅ Git operations work in Docker sandbox  
✅ Patch generated correctly (non-zero bytes)  
✅ Patch applies cleanly in evaluation sandbox  
✅ Tests pass using the patched code  
