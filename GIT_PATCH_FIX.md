# Git Patch Error Fix

## Problem
Evaluation was failing with error:
```
error: tests.py: already exists in working directory
error: main.py: already exists in working directory
error: beer_song.py: already exists in working directory
```

This happens when `git apply` receives a patch that tries to create files as "new file mode" when they already exist in the working directory.

## Root Cause

The `get_git_diff_helper()` function had a fallback that created patches with `new file mode 100644` headers, which tells git to CREATE new files. But these files already exist in the evaluation sandbox.

### Previous Code Issue:
```python
# OLD - Creates invalid patches
patch_lines.append(f"new file mode 100644")  # ❌ Wrong!
patch_lines.append(f"--- /dev/null")         # ❌ Tries to add as new file
```

## Solution

### 1. Removed Invalid Fallback
Removed the code that created `new file mode` patches manually. Now only uses proper `git diff --cached`.

### 2. Improved Git Staging
```python
# Stage modifications to tracked files
git add -u *.py

# Stage new files explicitly (excluding tests.py)
git ls-files --others --exclude-standard *.py
```

### 3. Added Debug Output
```python
debug_print(f"[GIT] Git status before staging:\n{status}")
debug_print(f"[GIT] Adding new files: {new_files}")
debug_print(f"[GIT] Generated patch ({len(patch)} bytes)")
```

## How It Works Now

### Agent Sandbox:
1. Starts with `main.py` (skeleton) + `tests.py` committed in git
2. Agent modifies `main.py` or creates new files
3. Git staging:
   - **Modified files**: `git add -u *.py` (stages changes to tracked files)
   - **New files**: `git add <file>` (stages untracked files, excluding tests.py)
4. Git diff: `git diff --cached HEAD` (shows only staged changes)
5. Result: Clean patch with modifications and/or additions

### Evaluation Sandbox:
1. Starts with same `main.py` (skeleton) + `tests.py` committed
2. Receives patch from agent
3. Validates: `git apply --check patch.diff`
4. Applies: `git apply patch.diff`
5. Runs tests against modified code

## Expected Patch Format

### For Modified Files (main.py):
```diff
diff --git a/main.py b/main.py
index abc123..def456 100644
--- a/main.py
+++ b/main.py
@@ -1,3 +1,10 @@
-def recite(start: int, take: int = 1) -> list[str]:
-    pass
+def recite(start: int, take: int = 1) -> list[str]:
+    # Implementation here
+    return result
```

### For New Files (beer_song.py):
```diff
diff --git a/beer_song.py b/beer_song.py
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/beer_song.py
@@ -0,0 +1,10 @@
+def verse(n: int) -> str:
+    # Implementation
+    return result
```

## Testing

Enable debug mode to see git operations:
```bash
DEBUG_MODE=true python ridges.py test-agent beer-song test_driven_agent_oop.py
```

Look for:
- `[GIT] Git status before staging:` - Shows file states
- `[GIT] Adding new files:` - Shows which new files are being added
- `[GIT] Generated patch (N bytes)` - Confirms patch was created

## Result

✅ Patches now correctly represent changes (modifications vs additions)  
✅ No more "already exists in working directory" errors  
✅ Evaluation sandbox can apply patches cleanly  
✅ Tests run against the correct modified code  
