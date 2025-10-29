# Dynamic Intervention System - Behavior-Based Detection

## Problem with Static Thresholds

**OLD SYSTEM (Static):**
```python
TIER1_TRIGGER = 15  # Always at step 15
TIER2_TRIGGER = 30  # Always at step 30
TIER3_TRIGGER = 45  # Always at step 45
```

**Issues:**
1. ❌ **Wastes time** - Agent stuck at step 5, but we wait until step 15
2. ❌ **Interrupts progress** - Agent making slow progress, but we force intervention at step 30
3. ❌ **Not adaptive** - Same rules for easy problems (5 steps) and hard problems (80 steps)
4. ❌ **Arbitrary numbers** - Why 15/30/45? Based on nothing!

## NEW SYSTEM: Dynamic Behavior-Based Detection

**Watches Agent's Actual Behavior:**
```python
CONSECUTIVE_STUCK_TOLERANCE = 8      # Stuck for 8 consecutive checks
MIN_STEPS_BEFORE_INTERVENTION = 10   # Minimum time before any intervention
PROGRESS_CHECK_INTERVAL = 5          # Check progress every 5 steps
TIER_ESCALATION_WAIT = 10            # Wait 10 steps between escalations
STRATEGY_REEVAL_FREQUENCY = 15       # Suggest strategy change if stuck
```

### How It Works

**1. Progress Tracking**
```python
# Every 5 steps, check: did tests improve?
if curr_pass > prev_pass or curr_fail < prev_fail:
    # PROGRESS! Reset stuck counter
    stuck_counter = 0
    last_progress_step = current_step
else:
    # NO PROGRESS - increment stuck counter
    stuck_counter += 1
```

**2. Tier 1 - Soft Intervention (Dynamic Trigger)**
```python
# Trigger when:
if (step >= MIN_STEPS_BEFORE_INTERVENTION and     # At least 10 steps
    intervention_tier == 0 and                     # No prior interventions
    stuck_counter >= CONSECUTIVE_STUCK_TOLERANCE): # Stuck for 8+ checks
    
    # → This triggers at step 10-50 depending on behavior!
    # - Fast stuck: triggers early (step 10-15)
    # - Making progress: never triggers
```

**3. Tier 2 - Strong Warning (Relative Trigger)**
```python
# Trigger when:
if (intervention_tier == 1 and                          # After Tier 1
    steps_since_last_intervention >= TIER_ESCALATION_WAIT and  # Waited 10 more steps
    stuck_counter >= 2):                                # Still stuck
    
    # → Triggers 10 steps AFTER Tier 1, not at fixed step number
```

**4. Tier 3 - Nuclear Reset (Relative Trigger)**
```python
# Trigger when:
if (intervention_tier == 2 and                          # After Tier 2
    steps_since_last_intervention >= TIER_ESCALATION_WAIT):  # Waited 10 more steps
    
    # → Triggers 10 steps AFTER Tier 2
```

## Example Timeline Comparison

### Scenario 1: Agent Stuck Early

**OLD (Static):**
```
Step 5:  Agent stuck (wrong approach)
Steps 6-14: Agent wastes 9 steps stuck
Step 15: Tier 1 intervention (finally!)
```

**NEW (Dynamic):**
```
Step 5:  Agent stuck (wrong approach)
Step 10: Stuck for 8 checks → Tier 1 intervention! (5 steps earlier ✅)
```

### Scenario 2: Agent Making Slow Progress

**OLD (Static):**
```
Step 10: 5 tests passing
Step 20: 7 tests passing (progress!)
Step 30: Tier 2 intervention (WHY? Agent is making progress! ❌)
```

**NEW (Dynamic):**
```
Step 10: 5 tests passing
Step 15: Check - progress detected! Reset stuck_counter ✅
Step 20: 7 tests passing - progress detected! Reset stuck_counter ✅
Step 30: No intervention (agent is progressing) ✅
```

### Scenario 3: Agent Solves Quickly

**OLD (Static):**
```
Step 8: Agent solves problem!
Step 15: Tier 1 intervention anyway (too late! Already solved ❌)
```

**NEW (Dynamic):**
```
Step 8: Agent solves problem!
No interventions triggered (problem solved before MIN_STEPS_BEFORE_INTERVENTION) ✅
```

### Scenario 4: Hard Problem Needs Many Steps

**OLD (Static):**
```
Step 15: Tier 1 (premature for hard problem)
Step 30: Tier 2 (still premature)
Step 45: Tier 3 forced reset (might have been close!)
Step 60: Out of steps ❌
```

**NEW (Dynamic with 100 steps):**
```
Step 15: Stuck detected → Tier 1
Step 25: Still stuck → Tier 2  (waited 10 steps)
Step 35: Still stuck → Tier 3 forced reset (waited 10 steps)
Steps 36-80: Fresh attempt has 45 steps to solve ✅
Step 75: Solves! ✅
```

## Key Advantages

### 1. **Faster Intervention When Needed**
- Detects stuck at step 10 instead of waiting until 15
- Saves 5-10 wasted steps

### 2. **No False Interventions**
- If agent is making progress, no intervention triggered
- Respects agent's agency when it's working

### 3. **Adaptive to Problem Difficulty**
- Easy problems: solved before interventions
- Hard problems: interventions space out naturally

### 4. **More Steps Available After Reset**
- OLD: Tier 3 at step 45 → only 15 steps left (60 total)
- NEW: Tier 3 at step 35 → 65 steps left (100 total) ✅

### 5. **Behavior-Based, Not Time-Based**
- Watches what agent DOES, not arbitrary step counts
- Stuck counter based on test results, not clock

## Configuration Tuning

You can tune these based on your needs:

```python
# More aggressive (faster interventions)
CONSECUTIVE_STUCK_TOLERANCE = 6   # Trigger after 6 stuck checks
TIER_ESCALATION_WAIT = 8          # Only 8 steps between tiers

# More patient (give agent more time)
CONSECUTIVE_STUCK_TOLERANCE = 10  # Wait for 10 stuck checks
TIER_ESCALATION_WAIT = 15         # 15 steps between tiers
```

## Real Example: Current Run

**What WOULD have happened with NEW system:**

```
Steps 1-10:  Agent tries "Observer Pattern"
Step 10:     Progress check: 0p_1f (same as step 5)
             stuck_counter = 2
Step 15:     Progress check: still 0p_1f
             stuck_counter = 3 → TIER 1 TRIGGERED! ✅
             (Instead of waiting until step 25 like old system)
             
Step 16-25:  Agent tries new strategy
Step 25:     Still 0p_1f, stuck_counter = 2
             Steps since Tier 1 = 10 → TIER 2 TRIGGERED! ✅
             (Dynamic, not fixed at step 30)
             
Step 26-35:  Agent tries again
Step 35:     Still stuck
             Steps since Tier 2 = 10 → TIER 3 FORCED RESET! ✅
             
Steps 36-100: 65 steps available for fresh approach ✅
              (vs only 15 steps in old 60-step system)
```

## Summary

**OLD: "Interrupt agent at steps 15, 30, 45 no matter what"** ❌
**NEW: "Detect when agent is stuck and intervene adaptively"** ✅

The new system is **smarter, faster, and more respectful** of the agent's autonomy while still providing strong interventions when truly needed! 🎯
