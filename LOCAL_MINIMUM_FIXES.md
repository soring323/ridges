# Local Minimum Problem - Fixes Implemented

## Problem Summary

The agent was getting stuck in **local minima** - spending 60 steps trying variations of the same failed approach instead of exploring fundamentally different solutions. This is a **Depth-First Search (DFS)** problem when **Breadth-First Search (BFS)** is needed.

### Evidence from Logs
```
Steps 35-60: "I've been stuck on this failing test for too long..."
             (Repeated 25+ times with same approach)
Step 60/60:  Max steps reached
Result:      Patch is None ❌
```

## Root Causes Identified

| Issue | Impact | Solution Implemented |
|-------|--------|---------------------|
| Strategy locked in early | Never reconsidered after step 1 | ✅ Strategy re-evaluation at checkpoints |
| No stuck detection | Agent unaware it's repeating itself | ✅ Progress monitoring every 5 steps |
| No escape mechanism | Has `start_over` tool but doesn't use it | ✅ Enhanced tool docs + intervention prompts |
| DFS problem | Goes deep into one approach | ✅ Escape hatch forces BFS exploration |

---

## Fixes Implemented

### Fix 1: Progress Monitoring System

**Location:** Lines 4532-4543 in `sam_fix.py`

```python
# Progress monitoring to detect stuck states
progress_tracker = {
    'last_test_results': [],      # Track test outcomes over time
    'stuck_counter': 0,            # Count consecutive identical results
    'approaches_tried': set(),     # Track which approaches were attempted
    'last_strategy_name': strategy.get('name'),
    'strategy_change_count': 0
}

# Checkpoints for interventions
STRATEGY_REEVAL_CHECKPOINTS = [20, 40]  # Re-evaluate strategy
TEST_CHECK_INTERVAL = 5                  # Check progress every 5 steps
```

**What it does:**
- Tracks test results every 5 steps
- Detects when results are identical (no progress)
- Maintains history of approaches tried

---

### Fix 2: Stuck Detection & Escape Hatch

**Location:** Lines 4571-4653 in `sam_fix.py`

**Mechanism:**
1. **Every 5 steps:** Run tests, extract pass/fail counts
2. **Compare results:** Check last 3 results (15 steps)
3. **Detect stuck:** If identical → increment stuck_counter
4. **Trigger escape:** After 3 consecutive identical results (15 steps with no progress)

**Intervention When Stuck:**
```python
if progress_tracker['stuck_counter'] >= 3:
    # Add critical intervention prompt
    intervention_prompt = """
🚨 CRITICAL INTERVENTION 🚨

You have been stuck for 15 steps with no progress (same test results).
Your current approach is NOT WORKING and you are stuck in a LOCAL MINIMUM.

REQUIRED ACTIONS:
1. STOP what you're doing - your current method has FAILED repeatedly
2. Call the 'start_over' tool to revert ALL your changes
3. Choose a COMPLETELY DIFFERENT architectural approach:
   - If you tried fixing existing code → Rewrite from scratch
   - If you tried single-phase updates → Try two-phase propagation
   - If you tried Observer pattern → Try Event queue system

DO NOT:
- Continue debugging the same code section
- Make small variations of the same fix

YOU MUST CHANGE YOUR FUNDAMENTAL APPROACH NOW.
"""
```

**Expected Behavior:**
- **Before fix:** Agent repeats same fix 40+ times, never escapes
- **After fix:** Agent gets strong intervention at step 15, forced to call `start_over`

---

### Fix 3: Strategy Re-Evaluation at Checkpoints

**Location:** Lines 4661-4693 in `sam_fix.py`

**Checkpoints:** Steps 20 and 40

**Process:**
1. Run tests to check if problem is solved
2. If still failing, note that current strategy hasn't worked
3. Add re-evaluation prompt:

```python
strategy_change_prompt = f"""
📊 STRATEGY RE-EVALUATION (Step {step}/60)

Your current strategy '{strategy_name}' has not succeeded after {step} steps.
Approaches already tried: {', '.join(approaches_tried)}

Consider a FUNDAMENTALLY DIFFERENT approach you haven't tried yet.
Review the test failures and choose a new architectural direction.
"""
```

**Purpose:** Give agent explicit checkpoints to reconsider its approach, even if not completely stuck.

---

### Fix 4: Enhanced `start_over` Tool Documentation

**Location:** Lines 2404-2432 in `sam_fix.py`

**Changes:**
- Added emoji indicator: 🔄 ESCAPE HATCH
- Explicit "STUCK IN LOCAL MINIMUM" language
- Clear when-to-use checklist with ✅/❌ indicators
- BFS vs DFS explanation
- Concrete examples of fundamental vs incremental changes

**Before:**
```python
'''
This will revert any changes made to the codebase and let's you start over.
Arguments: ...
'''
```

**After:**
```python
'''
🔄 ESCAPE HATCH: Revert ALL changes and start fresh with a different approach.

Use this tool when you're STUCK IN A LOCAL MINIMUM:
- Same test(s) failing for 10+ consecutive steps despite multiple fix attempts
- You've tried variations of the same approach without success
- You recognize your current architecture is fundamentally wrong
- Tests suggest you need a completely different design

This is a BREADTH-FIRST search tool - use it to explore different solution branches...

When to use:
✅ Test X fails repeatedly after trying 5+ different fixes to the same code section
✅ System intervention tells you to start over

When NOT to use:
❌ After only 1-2 fix attempts
❌ When tests are making progress

Example: If you tried fixing value comparison in a single-phase update system, 
try implementing a two-phase propagation system instead.
'''
```

---

## How It Works - Complete Flow

### Normal Execution (Making Progress)
```
Step 1-4:   Agent tries approach A
Step 5:     ✓ Progress check - some tests passing
Step 6-9:   Continue with approach A
Step 10:    ✓ Progress check - more tests passing
...
Step 30:    All tests pass ✅
```

### Stuck Detection Scenario (The Fix)
```
Step 1-4:   Agent tries approach A
Step 5:     Progress check - 2 tests failing
Step 6-9:   Agent tweaks approach A
Step 10:    Progress check - STILL 2 tests failing (stuck_counter = 1)
Step 11-14: Agent debugs approach A
Step 15:    Progress check - STILL 2 tests failing (stuck_counter = 2)
Step 16-19: Agent adds logging to approach A
Step 20:    🔄 CHECKPOINT - Strategy re-evaluation prompt added
            Progress check - STILL 2 tests failing (stuck_counter = 3)
            🚨 ESCAPE HATCH TRIGGERED!
            Critical intervention prompt injected
            
Step 21:    Agent receives intervention → calls start_over()
Step 22:    Codebase reverted, tries approach B (two-phase propagation)
Step 25:    Progress check - 1 test failing (progress made!)
Step 30:    All tests pass ✅
```

### Before Fix (Old Behavior)
```
Step 1-60:  Agent tries variations of approach A
            "I've been stuck..."
            "Let me try..."
            "I've been stuck..." (repeated 40 times)
Step 60:    Max steps → FAIL, Patch is None
```

---

## Testing the Fixes

### Expected Log Patterns

**Successful Intervention:**
```
[FIX_WORKFLOW] Progress checkpoint at step 15
[FIX_WORKFLOW] ⚠️  Stuck detected! Same test results for 15 steps: 2p_2f
[FIX_WORKFLOW] 🚨 Agent stuck in local minimum for 15 steps!
[FIX_WORKFLOW] Triggering escape hatch: Adding strong intervention prompt
[FIX_WORKFLOW] Intervention prompt added, proceeding to LLM call
[FIX_WORKFLOW] Step 16: Got tool 'start_over'  ← Agent responds to intervention!
[FIX_WORKFLOW] Step 17: Got tool 'apply_code_edit'  ← New approach being tried
[FIX_WORKFLOW] Progress checkpoint at step 20
[FIX_WORKFLOW] ✓ Progress made: 2p_2f → 3p_1f
```

**Strategy Re-evaluation:**
```
[FIX_WORKFLOW] 🔄 Strategy re-evaluation checkpoint at step 20
[FIX_WORKFLOW] Tests still failing at checkpoint 20, considering strategy change...
[FIX_WORKFLOW] Added strategy re-evaluation prompt
[FIX_WORKFLOW] Step 21: Got tool 'run_repo_tests'
[FIX_WORKFLOW] Step 22: Got tool 'apply_code_edit'  ← Different approach
```

### Metrics to Track

| Metric | Before Fix | After Fix (Expected) |
|--------|------------|---------------------|
| **Max steps hit** | 100% (always hits 60) | <30% (solves before limit) |
| **Average steps to solve** | 60 (fails) | 25-35 (succeeds) |
| **start_over calls** | 0 per run | 1-2 per run |
| **Approaches tried** | 1 (stuck in one) | 2-3 (explores multiple) |
| **Success rate** | Low (no patch) | High (generates patch) |

---

## Configuration Parameters

You can tune the intervention sensitivity:

```python
# In sam_fix.py lines 4542-4543
STRATEGY_REEVAL_CHECKPOINTS = [20, 40]  # When to force strategy reconsideration
TEST_CHECK_INTERVAL = 5                  # How often to check progress

# Stuck detection threshold (line 4597)
if progress_tracker['stuck_counter'] >= 3:  # 3 * 5 = 15 steps with no progress
```

**More aggressive (earlier intervention):**
```python
STRATEGY_REEVAL_CHECKPOINTS = [15, 30]
TEST_CHECK_INTERVAL = 3
stuck_threshold = 2  # Trigger after 6 steps
```

**More lenient (give agent more time):**
```python
STRATEGY_REEVAL_CHECKPOINTS = [25, 50]
TEST_CHECK_INTERVAL = 7
stuck_threshold = 4  # Trigger after 28 steps
```

---

## Key Insights

### The DFS vs BFS Problem

**DFS (Current Agent Behavior):**
```
Problem
  ↓
Fix comparison logic
  ↓
  Add None check
    ↓
    Try different operator
      ↓
      Add debug logging
        ↓
        (Goes infinitely deep in one branch)
```

**BFS (What We Want):**
```
Problem
  ↓
├─ Fix comparison logic (try 5 steps) → Fails
├─ Two-phase propagation (try 5 steps) → Works! ✅
├─ Event queue system (not tried)
└─ Topological sort (not tried)
```

**The fixes enable BFS by:**
1. **Detecting** when agent is going too deep (stuck detection)
2. **Interrupting** the deep dive (escape hatch)
3. **Forcing** consideration of alternatives (strategy re-evaluation)
4. **Providing tools** to switch branches (start_over)

---

## Summary

**What was broken:**
- Agent locked into one strategy
- No awareness of being stuck
- DFS problem-solving approach
- Never used `start_over` tool

**What was fixed:**
- ✅ Progress monitoring every 5 steps
- ✅ Stuck detection after 15 steps of no progress
- ✅ Automatic intervention with escape hatch
- ✅ Strategy re-evaluation at steps 20 & 40
- ✅ Enhanced tool documentation with explicit BFS guidance

**Expected improvement:**
- Agent will try 2-3 fundamentally different approaches instead of 1
- Success rate should increase significantly
- Will use `start_over` tool when appropriate
- Should solve problems in 25-35 steps instead of failing at 60

**Next steps:**
1. Test on the "react" problem again
2. Monitor logs for intervention triggers
3. Verify agent calls `start_over` when stuck
4. Tune parameters if needed

The local minimum trap has been addressed! 🎯
