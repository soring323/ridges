# Test Generation Configuration Guide

## Problem You Identified

```
[TEST_GEN] Gen 1: Found 13 test functions  ← Different every time!
[TEST_GEN] Gen 2: Found 9 test functions
[TEST_GEN] Gen 3: Found 12 test functions
[TEST_GEN] Consensus: appeared 1/5 times    ← Only 20% agreement = BAD!
```

**Issue**: Too much variation, no real consensus.

---

## Solution: Control Temperature

### Environment Variables

```bash
# 1. Test Generation Temperature (MOST IMPORTANT!)
export TEST_GEN_TEMPERATURE=0.0   # Deterministic (recommended)
export TEST_GEN_TEMPERATURE=0.3   # Diverse but inconsistent
export TEST_GEN_TEMPERATURE=0.7   # Very random (not recommended)

# 2. Number of Generations
export NUM_TEST_GENERATIONS=5     # Default: 5 times

# 3. Enable/Disable Consensus
export ENABLE_TEST_CONSENSUS=true # Default: enabled
```

---

## Recommended Settings

### Option 1: Deterministic (Recommended for Correctness)
```bash
export TEST_GEN_TEMPERATURE=0.0
export NUM_TEST_GENERATIONS=3
export ENABLE_TEST_CONSENSUS=true
```

**Expected Output:**
```
[TEST_GEN] Generation 1/3...
[TEST_GEN] Gen 1: Found 11 test functions
[TEST_GEN] Generation 2/3...
[TEST_GEN] Gen 2: Found 11 test functions  ← SAME!
[TEST_GEN] Generation 3/3...
[TEST_GEN] Gen 3: Found 11 test functions  ← SAME!
[TEST_GEN] ✓ Strong consensus (100% >= 80%)  ← PERFECT!
```

### Option 2: Diverse (If You Want Exploration)
```bash
export TEST_GEN_TEMPERATURE=0.2
export NUM_TEST_GENERATIONS=5
export ENABLE_TEST_CONSENSUS=true
```

**Expected Output:**
```
[TEST_GEN] Gen 1: Found 11 test functions
[TEST_GEN] Gen 2: Found 11 test functions
[TEST_GEN] Gen 3: Found 10 test functions
[TEST_GEN] Gen 4: Found 11 test functions
[TEST_GEN] Gen 5: Found 11 test functions
[TEST_GEN] ✓ Strong consensus (80% >= 80%)  ← GOOD!
```

### Option 3: Fast (Single Generation, No Consensus)
```bash
export ENABLE_TEST_CONSENSUS=false
```

---

## Validation Improvements

### Two-Step Validation Process:

**Step 1: Consensus** (Statistical)
- Generate tests N times
- Pick most common structure
- Eliminates random hallucinations

**Step 2: Correctness Validation** (Semantic)
- Validates tests against problem statement
- Checks if test inputs/outputs match examples
- Verifies all edge cases are covered
- Fixes incorrect expected values

---

## Consensus Quality Metrics

The agent now reports consensus strength:

```
[TEST_GEN] Consensus analysis:
  - Signature: ('test_basic', 'test_edge_zero', ...)
  - Appeared: 4/5 times (80%)
  - ✓ Strong consensus (80% >= 80%)
```

**Quality Levels:**
- **Strong (≥80%)**: High confidence, tests are reliable
- **Moderate (60-79%)**: Acceptable, but verify carefully
- **Weak (<60%)**: Low confidence, reduce temperature!

---

## How to Verify Tests Are Correct?

### Automatic Validation (Built-in)

The validation step checks:
1. ✅ Test inputs/outputs match problem examples
2. ✅ Edge cases from problem statement included
3. ✅ Boundary conditions tested
4. ✅ All function requirements covered
5. ✅ No incorrect expected values

### Manual Verification (Recommended)

After generation, check the tests:
```bash
# Look at generated tests
cat /tmp/task_*/tests.py

# Manually verify:
# 1. Do test values match problem statement examples?
# 2. Are edge cases mentioned in problem included?
# 3. Do assertions match expected outputs from problem?
```

---

## Quick Start

For **maximum correctness** (recommended):
```bash
export TEST_GEN_TEMPERATURE=0.0
export NUM_TEST_GENERATIONS=3
python ridges.py test-agent beer-song test_driven_agent_oop.py --timeout 50000
```

This will give you:
- 100% consensus (all 3 generations identical)
- Validated against problem requirements
- High confidence in test correctness

---

## Example: Good vs Bad

### ❌ Bad (Your Current Output)
```
Temperature: 0.3 (too high)
Gen 1: 13 tests
Gen 2: 9 tests   ← Different!
Gen 3: 12 tests  ← Different!
Consensus: 1/5 (20%)  ← No agreement!
```

### ✅ Good (With temperature=0.0)
```
Temperature: 0.0 (deterministic)
Gen 1: 11 tests
Gen 2: 11 tests  ← SAME!
Gen 3: 11 tests  ← SAME!
Consensus: 3/3 (100%)  ← Perfect agreement!
```

---

## The "Exercism Trick" 🎭

Their prompt includes this example format:
```python
# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/.../canonical-data.json
# File last updated on 2023-07-19
```

**This is pure prompt engineering!** The LLM:
- Mimics professional test file style
- Adds authoritative-looking comments
- Creates consistency across all test files
- **Doesn't actually fetch from GitHub** (it's just style!)

We've adopted a similar approach (without fake URLs):
```python
# These tests are auto-generated based on:
# Problem specification and canonical test patterns
```

This makes tests look more professional and structured.

---

## Summary

**To answer your questions:**

1. **How to be confident tests are correct?**
   - Use temperature=0.0 for consistency
   - Check consensus strength (aim for ≥80%)
   - Validation step checks against problem requirements
   - Manually verify critical test cases

2. **How to control temperature?**
   - Set `TEST_GEN_TEMPERATURE=0.0` environment variable
   - Lower = more consistent, higher = more diverse
   - Recommended: 0.0 for production, 0.2 for exploration

3. **How do they reference Exercism?**
   - It's in the prompt as an **example format** to copy
   - The LLM mimics the style (including comments)
   - No actual GitHub fetching happens
   - Pure psychological trick to make tests look canonical!
